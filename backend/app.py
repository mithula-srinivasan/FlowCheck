from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import io, numpy as np, webrtcvad
import tempfile, soundfile as sf
import whisper
import joblib
import os

from utils.extract_features import extract_audio_features
from utils.synonym_suggester import get_synonyms

whisper_model = whisper.load_model("tiny")
ml_model = joblib.load("model/stuttering_model.pkl")

app = FastAPI(title="FlowCheck API")

app.add_middleware(
    CORSMiddleware, # cross origin resource sharing
    allow_origins=["*"],  # allows all websites
    allow_credentials=True, # allows cookies/credentials
    allow_methods=["*"], # allows HTTP methods, get post put delet etc
    allow_headers=["*"], # allow all request headers
)

class PauseSegment(BaseModel):
    start_s: float
    end_s: float
    duration_ms: float
    label: str = "unknown" # placeholder: naturalsentence/midsentence (later)

class VoicedSegment(BaseModel):
    start_s: float
    end_s: float
    duration_ms: float

class Word(BaseModel):
    word: str
    start: float
    end: float

class WhisperSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] | None = None

class AnalysisResult(BaseModel):
    duration_sec: float
    voiced_ratio: float
    avg_pause_ms: float
    longest_pause_ms: float
    blocks: int
    pauses: list[PauseSegment]
    voiced_segments: list[VoicedSegment]
    transcript: str | None = None
    segments: list[WhisperSegment] | None = None
    stammer_score: float | None = None
    fluency_level: str | None = None
    stutter_detected: bool | None = None
    ml_probability: float | None = None
    word_suggestions: dict[str, list[str]] | None = None



@app.get("/health")
def health():
    return {"ok": True}


def convert_to_mono16k(raw_bytes: bytes): # raw_bytes - raw binary data from audio file
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), always_2d=False)
        # data holds a NumPy array of amplitude samples
        # sr - sample rate
        # always2d = mono - 1d array, 2 channels 2d array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")
    
    if data.ndim > 1: # if audio has multiple channels, take mean 
        data = np.mean(data, axis=1) # take mean to converge to single mono waveform

    if sr!=16000:
        duration = len(data) / sr
        target_len = int(duration * 16000)
        x_old = np.linspace(0, 1, num=len(data), endpoint=False)
        x_new = np.linspace(0, 1, num=target_len, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
        sr = 16000
    return np.clip(data, -1.0, 1.0).astype(np.float32), sr

def vad_speech_detection(y: np.ndarray, sr: int, frame_ms: int = 20, aggression: int = 2):
    vad = webrtcvad.Vad(aggression)
    # aggression: how strict it is, 2 is middle ground
    frame_len = int(sr * frame_ms / 1000)
    if frame_len==0 or len(y) < frame_len:
        return [], frame_ms
    pcm16 = (y * 32767).astype(np.int16).tobytes()
    frames = [pcm16[i:i+2*frame_len] for i in range(0,len(pcm16), 2*frame_len)]
    flags=[]
    for f in frames:
        if len(f) < 2*frame_len: break
        flags.append(vad.is_speech(f,sr))
    return flags, frame_ms

def contiguous_regions(flags: list[bool], frame_ms: int, sr: int, voiced=True):
    segs = []
    i = 0
    n = len(flags)
    
    while i < n:
        if flags[i] == voiced:
            j = i
            while j < n and flags[j] == voiced:
                j+=1
            start_s = i * frame_ms / 1000.0
            end_s = j * frame_ms / 1000.0
            segs.append((start_s, end_s))
            i = j
        else: 
            i+=1
    return segs

def calculate_fluency_score(voiced_ratio, avg_pause_ms, longest_pause_ms, blocks):
    score = 100.0
    if voiced_ratio < 0.6:
        score -= (0.6 - voiced_ratio) * 100
    if avg_pause_ms > 300:
        score -= (avg_pause_ms - 300) / 10
    if longest_pause_ms > 500:
        score -= (longest_pause_ms - 500) / 20
    if blocks > 10:
        score -= (blocks - 10) * 2
    score = max(0, min(100, round(score, 1)))
    if score >= 85:
        level = "Excellent"
    elif score >= 70:
        level = "Moderate"
    elif score >= 50:
        level = "Mild Disfluency"
    elif score >= 30:
        level = "Significant disfluency"
    else:
        level = "Severe disfluency"

    return score, level


def analyze_audio(y, sr):
    """Analyze audio for pauses, voiced segments, and fluency score."""
    flags, frame_ms = vad_speech_detection(y, sr, frame_ms=20, aggression=2)
    if not flags:
        raise HTTPException(status_code=400, detail="Could not compute VAD.")

    voiced_ratio = float(np.mean(flags))
    voiced_runs = contiguous_regions(flags, frame_ms, sr, voiced=True)
    silence_runs = contiguous_regions(flags, frame_ms, sr, voiced=False)

    pause_ms_list = [(end - start) * 1000.0 for (start, end) in silence_runs]
    avg_pause_ms = float(np.mean(pause_ms_list)) if pause_ms_list else 0.0
    longest_pause_ms = float(np.max(pause_ms_list)) if pause_ms_list else 0.0
    blocks = sum(1 for p in pause_ms_list if p > 400.0)

    pauses = [
        PauseSegment(start_s=round(s,3), end_s=round(e,3),
                     duration_ms=round((e-s)*1000.0,1), label="unknown")
        for (s,e) in silence_runs
    ]

    voiced_segments = [
        VoicedSegment(start_s=round(s,3), end_s=round(e,3),
                      duration_ms=round((e-s)*1000.0,1))
        for (s,e) in voiced_runs
    ]

    stammer_score, fluency_level = calculate_fluency_score(
        voiced_ratio, avg_pause_ms, longest_pause_ms, blocks
    )

    return {
        "voiced_ratio": voiced_ratio,
        "avg_pause_ms": avg_pause_ms,
        "longest_pause_ms": longest_pause_ms,
        "blocks": blocks,
        "pauses": pauses,
        "voiced_segments": voiced_segments,
        "stammer_score": stammer_score,
        "fluency_level": fluency_level,
    }


def enhance_language(transcript: str, stammer_score: float):
    """Generate synonym suggestions if fluency is low."""
    if not transcript or stammer_score >= 80:
        return {}

    words = [w.strip(",.?!'\"").lower() for w in transcript.split()]
    suggestions = {w: get_synonyms(w) for w in set(words) if get_synonyms(w)}
    return {w: s for w, s in suggestions.items() if s}


def detect_long_duration_words(transcript_segments):
    """Detect words that took unusually long to say (likely stuttered)"""
    difficult_words = []
    print("Checking for long duration words...")

    for segment in transcript_segments:
        print(f"Segment: {segment['text']}")
        for word in segment.get("words", []):
            duration = word["end"] - word["start"]
            word_text = re.sub(r'[^\w\s]', '', word["word"]).strip().lower()
            print(f"   Word: '{word['word']}' -> '{word_text}' | Duration: {duration:.2f}s")

            if duration > 0.7 and len(word_text) > 3:
                difficult_words.append({
                    "word": word_text,
                    "original": word["word"],
                    "duration": duration,
                    "reason": f"long_duration_{duration:.1f}s"
                })

    print(f"Total long words found: {len(difficult_words)}")
    return difficult_words


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw: 
        raise HTTPException(status_code=400, detail="Empty file.")
    # raw contains audio bytes as a file
    y, sr = convert_to_mono16k(raw)
    duration_sec = len(y) / sr
    if duration_sec < 2.0:
        raise HTTPException(status_code=400, detail="Clip too short (<2s).")
    if duration_sec > 60.0:
        raise HTTPException(status_code=400, detail="Clip too long (>60s).")

    audio_result = analyze_audio(y, sr)

    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tmpfile:
        sf.write(tmpfile.name, y, sr)
        transcription = whisper_model.transcribe(tmpfile.name, word_timestamps=True)
        
        features = extract_audio_features(tmpfile.name)
        if features is not None:
            stutter_pred = bool(ml_model.predict([features])[0])
            stutter_prob = float(ml_model.predict_proba([features])[0][1])
        else:
            stutter_pred = None
            stutter_prob = None
    
    os.remove(tmpfile.name)

    transcript_text = transcription["text"]
    transcript_segments = transcription["segments"]

    flags, frame_ms = vad_speech_detection(y, sr, frame_ms=20, aggression=2)
    if not flags:
        raise HTTPException(status_code=400, detail="Could not compute VAD.")
    
    voiced_ratio = float(np.mean(flags))
    voiced_runs = contiguous_regions(flags,frame_ms,sr,voiced=True)
    silence_runs = contiguous_regions(flags,frame_ms,sr,voiced=False)

    pause_ms_list = [ (end - start) * 1000.0 for (start, end) in silence_runs ]
    avg_pause_ms = float(np.mean(pause_ms_list)) if pause_ms_list else 0.0
    longest_pause_ms = float(np.max(pause_ms_list)) if pause_ms_list else 0.0
    count = 0
    for p in pause_ms_list:
        if p > 400.0:
            count += 1
    blocks = count

    pauses = [
        PauseSegment(
            start_s=round(s,3), 
            end_s=round(e,3),
            duration_ms=round((e-s)*1000.0,1),
            label="unknown"
        )
        for(s,e) in silence_runs
    ]

    voiced_segments = [
        VoicedSegment(
            start_s=round(s,3), 
            end_s=round(e,3),
            duration_ms=round((e-s)*1000.0,1),
        )
        for(s,e) in voiced_runs
    ]

    stammer_score, fluency_level = calculate_fluency_score(voiced_ratio, avg_pause_ms, longest_pause_ms, blocks)
    
    difficult_words = []
    long_duration_detected = detect_long_duration_words(transcript_segments)
    if long_duration_detected:
        for item in long_duration_detected:
            difficult_words.append(item["word"])
    
    print(difficult_words)
    difficult_words = list(set(difficult_words))
    suggestions = {w: get_synonyms(w) for w in difficult_words}
    word_suggestions = {w: s for w, s in suggestions.items() if s and len(s) > 0}
    print(word_suggestions)


    return AnalysisResult(
        duration_sec=round(duration_sec,2),
        voiced_ratio=round(voiced_ratio,3),
        avg_pause_ms=round(avg_pause_ms,1),
        longest_pause_ms=round(longest_pause_ms,1),
        blocks=blocks,
        pauses=pauses,
        voiced_segments=voiced_segments,
        transcript = transcript_text,
        segments = transcript_segments,
        stammer_score=stammer_score,      
        fluency_level=fluency_level,
        stutter_detected = stutter_pred,
        ml_probability = round(stutter_prob,2) if stutter_prob else None,
        word_suggestions=word_suggestions
    )
    
