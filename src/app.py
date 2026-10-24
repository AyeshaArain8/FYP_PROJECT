"""
Two-way PSL communication app.

Left panel:  Sign -> Text / Voice  (upload a video of someone signing)
Right panel: Text -> Sign          (type a sentence, get a sign animation)

USAGE
    streamlit run src/app.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import speech_recognition as sr
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from inference_continuous import extract_video_landmarks, sliding_window_predict
from gloss_to_english import gloss_to_english
from tts import save_audio
from text_to_sign import render_sign_sequence
from model_io import load_trained_model

st.set_page_config(page_title="PSL Two-Way Communication", layout="wide", page_icon="🤟")

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
h1 { color: #6C2BD9; }
h2 { color: #6C2BD9; border-bottom: 2px solid #E9D8FD; padding-bottom: 8px; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FAFAFF;
    border-radius: 14px;
    padding: 8px;
}
.stButton > button {
    background: linear-gradient(90deg, #6C2BD9, #D6336C);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: 600;
}
.stButton > button:hover { opacity: 0.9; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🤟 Pakistan Sign Language — Two-Way Communication")
st.caption("Sign ↔ Text ↔ Voice, built on a 68-word PSL vocabulary.")


@st.cache_resource
def load_model_and_labels():
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    model = load_trained_model()
    return model, sorted(label_map["idx_to_class"])


model, classes = load_model_and_labels()

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.header("🎥 Sign → Text / Voice")
        st.write("Someone signs, you get English text + spoken audio.")

        mode = st.radio(
            "Input method",
            options=["📤 Upload Video", "📹 Live Camera"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if mode == "📤 Upload Video":
            uploaded_video = st.file_uploader(
                "Upload a video of someone signing", type=["mp4", "avi", "mov"]
            )
            if uploaded_video is not None:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(uploaded_video.read())
                    video_path = tmp.name

                with st.spinner("Decoding signs..."):
                    feats = extract_video_landmarks(video_path)
                    glosses = sliding_window_predict(model, classes, feats)
                    english = gloss_to_english(glosses)

                st.success(f"**Decoded signs:** {', '.join(glosses) if glosses else '(none detected)'}")
                st.info(f"**English:** {english or '—'}")

                if english:
                    audio_path = str(config.PLOTS_DIR / "output_speech.wav")
                    save_audio(english, audio_path)
                    st.audio(audio_path)

        else:  # Live Camera
            st.caption(
                "Opens a webcam window (needs its own window since live video "
                "requires continuous frame capture a browser page can't do "
                "alone). It waits for a **stable, high-confidence** prediction "
                "before accepting a word — signs won't be rushed. "
                "Speaks each confirmed word, builds a sentence on screen. "
                "Press **c** to clear, **q** to close."
            )
            if st.button("🎥 Start Live Camera", use_container_width=True):
                script_path = str(Path(__file__).resolve().parent / "predict_live_continuous.py")
                subprocess.Popen([sys.executable, script_path])
                st.success("Live camera window launched — check your taskbar.")

with col2:
    with st.container(border=True):
        st.header("✍️ Text → Sign")
        st.write("Give words (typed or spoken), get the matching PSL sign animation.")

        input_mode = st.radio(
            "Input method",
            options=["⌨️ Type/Select Words", "🎤 Speak"],
            horizontal=True,
            label_visibility="collapsed",
        )

        words_to_render = []

        if input_mode == "⌨️ Type/Select Words":
            selected_words = st.multiselect(
                "Choose words (in the order you want them signed)",
                options=classes,
                help="Only vocabulary the model was trained on can be animated.",
            )
            words_to_render = [w.lower() for w in selected_words]

        else:  # Speak
            st.caption(
                "Record yourself saying a sentence. Only words that match the "
                "68-word vocabulary will actually be signed -- anything else "
                "gets skipped and shown to you below."
            )
            audio_value = st.audio_input("Record your voice")

            if audio_value is not None:
                with st.spinner("Transcribing..."):
                    recognizer = sr.Recognizer()
                    try:
                        with sr.AudioFile(audio_value) as source:
                            audio_data = recognizer.record(source)
                        transcript = recognizer.recognize_google(audio_data)
                    except sr.UnknownValueError:
                        transcript = ""
                        st.warning("Couldn't understand the audio -- try speaking clearly, closer to the mic.")
                    except sr.RequestError:
                        transcript = ""
                        st.error("Speech recognition needs an internet connection -- check your connection and try again.")

                if transcript:
                    st.write(f"**Heard:** {transcript}")
                    heard_words = transcript.lower().split()
                    matched = [w for w in heard_words if w in classes]
                    missing = [w for w in heard_words if w not in classes]
                    if matched:
                        st.success(f"Matched vocabulary words: {', '.join(matched)}")
                    if missing:
                        st.warning(f"Not in vocabulary (skipped): {', '.join(missing)}")
                    words_to_render = matched

        if st.button("🤟 Generate Sign Animation", use_container_width=True):
            if not words_to_render:
                st.warning("No valid vocabulary words to animate yet.")
            else:
                with st.spinner("Rendering sign animation..."):
                    gif_path = render_sign_sequence(words_to_render)
                if gif_path:
                    st.image(gif_path, caption=" → ".join(words_to_render))

        with st.expander(f"📖 Full vocabulary ({len(classes)} words)"):
            st.write(", ".join(classes))
