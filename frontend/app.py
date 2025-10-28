import streamlit as st
import requests
import io

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="FlowCheck", 
    page_icon="🎤", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main {
        background-color: #F5F5F5;
    }
    .sage-bg {
        background-color: #9CAF88;
        padding: 100px;
        border-radius: 20px;
        text-align: center;
    }
    .lilac-bg {
        background-color: #C8A2C8; 
        padding: 40px;
        border-radius: 15px;
    }
    .white-text {
        color: white;
    }
    .big-title {
        font-size: 80px;
        font-weight: bold;
    }
    .medium-title {
        font-size: 40px;
    }
</style>
""", unsafe_allow_html=True)


def show_landing():
    st.markdown("""
    <div class="sage-bg">
        <p class="white-text medium-title">Welcome to</p>
        <h1 class="white-text big-title">FlowCheck</h1>
        <p class="white-text medium-title">Speech Fluency Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Begin Your Journey", type="primary", use_container_width=True):
        st.session_state.page = "welcome"
        st.rerun()


def show_welcome():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #9CAF88; padding: 40px; border-radius: 15px; height: 500px;'>
            <h2 class="white-text">🗣️ Your Voice Matters</h2>
            <p class="white-text" style='font-size: 18px;'>
            At FlowCheck, we celebrate every voice exactly as it is. 
            Stuttering isn't a flaw - it's part of your unique speaking pattern.
            </p>
            <p class="white-text" style='font-size: 18px;'>
            No judgments. No pressure. Just supportive tools to help 
            you communicate with confidence.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #C8A2C8; padding: 40px; border-radius: 15px; height: 500px;'>
            <h2 class="white-text">🌟 Famous People Who Stuttered</h2>
            <ul class="white-text" style='font-size: 16px;'>
            <li>Winston Churchill - Leader</li>
            <li>Marilyn Monroe - Icon</li>
            <li>King George VI - Monarch</li>
            <li>Emily Blunt - Actress</li>
            <li>Joe Biden - President</li>
            <li>James Earl Jones - Voice Actor</li>
            </ul>
            <p class="white-text" style='font-size: 16px;'>
            Your voice can change the world too, exactly as it is.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("Start Analysis", type="primary", use_container_width=True):
        st.session_state.page = "main"
        st.rerun()


def show_main_app():
    st.title("🎯 FlowCheck - Speech Analysis")
    
    st.write("Upload your audio to analyze speech patterns and get helpful suggestions!")
    
    audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a'])
    
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        
        if st.button("Analyze Speech", type="primary"):
            with st.spinner("🔍 Analyzing your speech patterns..."):
                try:
                    files = {"file": audio_file}
                    response = requests.post(f"{BACKEND_URL}/analyze", files=files)
                    
                    if response.status_code == 200:
                        results = response.json()
                        display_results(results)
                    else:
                        st.error("Analysis failed. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error connecting to server: {e}")

def display_results(results):
    st.success("✅ Analysis Complete!")
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fluency Score", f"{results['stammer_score']}/100")
    with col2:
        st.metric("Fluency Level", results['fluency_level'])
    with col3:
        st.metric("Stutter Detected", "Yes" if results['stutter_detected'] else "No")
    
    if results.get('word_suggestions'):
        st.subheader("💡 Word Suggestions")
        for word, suggestions in results['word_suggestions'].items():
            st.info(f"You struggled with **'{word}'** - try: {', '.join(suggestions)}")
    
    if results.get('transcript'):
        st.subheader("📝 Transcription")
        st.write(results['transcript'])
    
    if st.button("Analyze Another Sample"):
        st.session_state.page = "main"
        st.rerun()

if 'page' not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    show_landing()
elif st.session_state.page == "welcome":
    show_welcome()
elif st.session_state.page == "main":
    show_main_app()