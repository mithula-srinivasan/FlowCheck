# FlowCheck - Stuttering Detection Web App

**FlowCheck** is an AI-driven speech fluency analysis tool designed to detect and evaluate stuttering patterns in recorded speech.  
It integrates **Whisper AI**, **audio signal processing**, and **machine learning** to provide fluency scoring, word-level analysis, and synonym suggestions for stuttered words, built with empathy for people who stammer.

---

## Personal Motivation

As someone who personally experienced stuttering during childhood, I wanted to create a project that resonated with that journey. Something that could help others facing similar challenges.  
FlowCheck is an independent project inspired by my personal struggle with fluency, evolving into a humane and technical solution.  

The current version runs locally as a web app, but my long-term goal is to deploy it publicly and expand it with features that support self-practice, and confidence-building for individuals who stutter.

---

## Technical Overview

### Backend (FastAPI)
- Developed a RESTful API using FastAPI, performing complete audio analysis:
  - **Whisper AI** for speech-to-text transcription with word-level timestamps  
  - **WebRTC VAD** (Voice Activity Detection) for silence and block segmentation  
  - **Audio feature extraction** using `librosa` (MFCC, ZCR, RMS, Spectral Centroid)  
  - **Machine learning model** (Random Forest Classifier) trained on stuttering events  
- Generates detailed results including fluency score, disfluency classification, and word analysis.

#### Dataset Usage
- The **machine learning model** in FlowCheck was trained using a **subset of the public Stuttering Events in Podcasts Dataset (SEP-28k)**, originally released by Apple researchers (*Lea et al., ICASSP 2021*).  
- Out of the full 28,000 three-second clips available in the dataset, **1,383 clips (approximately 10 hours of audio)** were selectively used for model training and evaluation.  
- The ML model was curated and preprocessed for:
  - **Segment-level feature extraction** (MFCC, ZCR, RMS, etc.)
  - **Balanced class distribution** between fluent and disfluent speech  
- Data stored under `/data/ml-stuttering-events-dataset`, used exclusively for local experimentation and model improvement.
- The dataset was used solely for academic and non-commercial purposes to train the Random Forest classifier and extract acoustic features such as **MFCC**, **ZCR**, **RMS**, and **Spectral Centroid**.  
- The SEP-28k dataset remains the property of its original creators and is licensed under **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)**.  



### Frontend (Streamlit)
- Built a clean, accessible UI in **Streamlit** with a **Sage Green and Lilac** theme.
- Allows users to:
  - Upload audio clips (WAV, MP3, M4A)
  - Receive fluency score and fluency level
  - View stutter detection probability
  - Get word replacement suggestions for smoother phrasing
  - See Whisper-generated transcript
- Designed for zero-judgment, supportive feedback for users who stammer.

### Core Functionalities
- Calculates fluency using voiced/silent segment ratios
- Identifies long duration and audio features based disfluencies  
- Suggests context-aware synonyms via NLTK WordNet  
- Works fully offline and no external API calls needed

---

## Tech Stack

**Languages & Frameworks:**  
Python, FastAPI, Streamlit  

**Libraries & Tools:**  
Whisper AI, Librosa, Scikit-Learn, NLTK, NumPy, SoundFile, Joblib, WebRTC VAD  

**Model:**  
Random Forest Classifier trained on custom stuttering event dataset  

**Deployment:**  
Local runtime (Uvicorn + Streamlit); cloud version in progress for Hugging Face Spaces  

---

## Acknowledgements

Special thanks to the open-source community behind OpenAI Whisper, Streamlit, and Scikit-Learn.  
FlowCheck is dedicated to everyone who’s ever paused, repeated, or struggled to find their voice and to remind them their voice matters.

