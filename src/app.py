
"""
Two-way PSL communication app.

Left panel:  Sign -> Text / Voice
Right panel: Text -> Sign

USAGE
    streamlit run src/app.py
"""

import json
import re
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


st.set_page_config(
    page_title="PSL Two-Way Communication",
    layout="wide",
    page_icon="🤟",
)

st.markdown("""
<style>
.stApp { background: #F5F8FC; color: #17212B; }
[data-testid="stHeader"] { background: rgba(245, 248, 252, 0.92); }
.block-container {
    max-width: 1280px;
    padding: 3.5rem clamp(1rem, 3vw, 2.25rem) 3rem;
}
.hero-kicker {
    margin: 0 0 0.2rem 0.15rem;
    color: #0E8290;
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
h1 {
    color: #17324D;
    font-size: clamp(1.7rem, 4vw, 2.55rem) !important;
    line-height: 1.15 !important;
    letter-spacing: -0.025em;
    margin-bottom: 0.25rem !important;
}
h2 {
    color: #155E75;
    border-bottom: 2px solid #D8E9EF;
    padding-bottom: 0.55rem;
    font-size: clamp(1.25rem, 2.2vw, 1.55rem) !important;
    line-height: 1.25 !important;
}
[data-testid="stCaptionContainer"] { color: #526476; }
[data-testid="stHorizontalBlock"] { gap: 1rem; }
[data-testid="stFileUploaderDropzone"] { border-radius: 14px; }
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label,
[data-testid="stRadio"] label *,
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label * {
    color: #17212B !important;
    opacity: 1 !important;
}
[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #17212B !important;
}
[data-baseweb="select"] *,
[data-testid="stFileUploaderDropzone"] * {
    color: #17212B !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #F1F5F9 !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE7EE;
    border-radius: 18px;
    padding: 12px;
    box-shadow: 0 8px 22px rgba(23, 50, 77, 0.06);
}
.stButton > button {
    background: linear-gradient(100deg, #0E8290, #155E75);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    padding: 0.65em 1.2em;
    font-weight: 600;
}
.stButton > button:hover { opacity: 0.92; color: white; border: none; }
[data-testid="stRadio"] div[role="radiogroup"] { gap: 0.5rem; }
img, video, canvas { max-width: 100%; }
@media (max-width: 680px) {
    .block-container { padding: 2.5rem 0.7rem 2rem; }
    h1 { font-size: 1.65rem !important; line-height: 1.16 !important; }
    h2 { font-size: 1.18rem !important; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.65rem; }
    [data-testid="column"] {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 100% !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.55rem;
        border-radius: 15px;
    }
    [data-testid="stRadio"] div[role="radiogroup"] {
        flex-wrap: wrap;
        column-gap: 0.4rem;
        row-gap: 0.2rem;
    }
    [data-testid="stRadio"] label { font-size: 0.9rem !important; }
    .stButton > button { width: 100%; min-height: 48px; }
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stTextInput"] input { min-height: 44px; }
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label * { color: #17212B !important; opacity: 1 !important; }
    [data-baseweb="select"] > div { background-color: #FFFFFF !important; }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-kicker">Pakistan Sign Language</div>', unsafe_allow_html=True)
st.title("Two-Way Communication")
st.caption("Sign ↔ English text and voice · 68-word PSL vocabulary")


def get_classes_in_model_order(label_map):
    """
    Return class names in exactly the same order used by model output indices.

    Supports:
    1. List format: ["hello", "water", ...]
    2. Dictionary format: {"0": "hello", "1": "water", ...}
    """

    idx_to_class = label_map["idx_to_class"]

    if isinstance(idx_to_class, list):
        classes = idx_to_class

    elif isinstance(idx_to_class, dict):
        try:
            ordered_items = sorted(
                idx_to_class.items(),
                key=lambda item: int(item[0])
            )
        except (TypeError, ValueError):
            raise ValueError(
                "Invalid label_map: idx_to_class dictionary keys "
                "must be numeric class indices."
            )

        indices = [int(index) for index, _ in ordered_items]

        expected_indices = list(range(len(ordered_items)))

        if indices != expected_indices:
            raise ValueError(
                "Invalid label_map: class indices must start at 0 "
                "and be continuous."
            )

        classes = [class_name for _, class_name in ordered_items]

    else:
        raise ValueError(
            "Invalid label_map: idx_to_class must be a list or dictionary."
        )

    if not classes:
        raise ValueError("No classes found in label_map.json.")

    return [str(class_name) for class_name in classes]


@st.cache_resource
def load_model_and_labels():
    with open(config.LABEL_MAP_PATH, encoding="utf-8") as f:
        label_map = json.load(f)

    # IMPORTANT:
    # Do not alphabetically sort these labels.
    # Model output index 0 must match class index 0,
    # model output index 1 must match class index 1, and so on.
    classes = get_classes_in_model_order(label_map)

    model = load_trained_model()

    # Check that model output count matches the label count.
    output_shape = model.output_shape

    if isinstance(output_shape, list):
        raise ValueError(
            "This app expects a model with one classification output."
        )

    model_output_classes = output_shape[-1]

    if model_output_classes is not None:
        if int(model_output_classes) != len(classes):
            raise ValueError(
                f"Model has {model_output_classes} output classes, "
                f"but label_map contains {len(classes)} classes. "
                "Check the model and label_map.json."
            )

    return model, classes


# Urdu meanings are for display only.
# Original English class labels remain unchanged for the model.
URDU_MEANINGS = {
    "absolutely": "بالکل",
    "aircrash": "ہوائی حادثہ",
    "airplane": "ہوائی جہاز",
    "all": "سب",
    "also": "بھی",
    "arrival": "آمد",
    "assalam-o-alaikum": "السلام علیکم",
    "atm": "اے ٹی ایم",
    "bald": "گنجا",
    "beach": "ساحل",
    "beak": "چونچ",
    "bear": "ریچھ",
    "beard": "داڑھی",
    "bed": "بستر",
    "bench": "بینچ",
    "bicycle": "سائیکل",
    "both": "دونوں",
    "bridge": "پل",
    "bring": "لانا",
    "bulb": "بلب",
    "cartoon": "کارٹون",
    "chimpanzee": "چمپینزی",
    "color_pencils": "رنگین پنسلیں",
    "cow": "گائے",
    "crow": "کوا",
    "cupboard": "الماری",
    "deer": "ہرن",
    "doctor": "ڈاکٹر",
    "dog": "کتا",
    "donttouch": "مت چھوئیں",
    "door": "دروازہ",
    "elephant": "ہاتھی",
    "excuseme": "معاف کیجیے",
    "facelotion": "فیس لوشن",
    "fan": "پنکھا",
    "garden": "باغ",
    "generator": "جنریٹر",
    "goodbye": "خدا حافظ",
    "goodmorning": "صبح بخیر",
    "have_a_good_day": "آپ کا دن اچھا گزرے",
    "hello": "سلام",
    "help": "مدد",
    "hungry": "بھوکا",
    "ihaveacomplaint": "مجھے شکایت ہے",
    "left_hand": "بایاں ہاتھ",
    "lifejacket": "لائف جیکٹ",
    "mine": "میرا",
    "mobile_phone": "موبائل فون",
    "nailcutter": "نیل کٹر",
    "need": "ضرورت",
    "nothing": "کچھ نہیں",
    "peacock": "مور",
    "policecar": "پولیس کی گاڑی",
    "razor": "استرا",
    "shampoo": "شیمپو",
    "shower": "شاور",
    "sunglasses": "دھوپ کا چشمہ",
    "thankyou": "شکریہ",
    "tissue": "ٹشو",
    "toothbrush": "دانت صاف کرنے کا برش",
    "toothpaste": "دانت صاف کرنے کا پیسٹ",
    "umbrella": "چھتری",
    "water": "پانی",
    "we": "ہم",
    "welldone": "شاباش",
    "where": "کہاں",
    "who": "کون",
    "you": "آپ",
}


model, classes = load_model_and_labels()

pipeline_feedback = st.session_state.pop("word_pipeline_feedback", None)
if pipeline_feedback:
    st.success(pipeline_feedback["message"])
    with st.expander("Training and evaluation log"):
        st.code(pipeline_feedback["log"] or "Completed successfully.")


def launch_word_recorder(word, sample_count):
    recorder = Path(__file__).resolve().parent / "record_word.py"
    subprocess.Popen(
        [sys.executable, str(recorder), "--word", word, "--samples", str(int(sample_count))],
        cwd=str(config.ROOT_DIR),
    )


def process_word_recordings(word, minimum):
    pipeline = Path(__file__).resolve().parent / "word_pipeline.py"
    with st.spinner("Adding examples and training the PSL model. This can take several minutes..."):
        result = subprocess.run(
            [sys.executable, str(pipeline), "--word", word, "--minimum", str(minimum)],
            cwd=str(config.ROOT_DIR), text=True, capture_output=True,
        )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode:
        st.error("Training did not finish. If processing had started, the saved model/data snapshot was restored.")
        st.code(output[-12000:] or f"Exit code: {result.returncode}")
        return
    st.session_state["word_pipeline_feedback"] = {
        "message": f"'{word}' was added and the model/templates were updated.",
        "log": output[-12000:],
    }
    st.cache_resource.clear()
    st.rerun()

# Display labels map back to the unchanged model class names.
display_to_class = {
    f"{word} — {URDU_MEANINGS.get(word, 'اردو معنی دستیاب نہیں')}": word
    for word in classes
}

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
            st.caption("Video is split at short no-hand gaps. Low-confidence guesses are skipped to reduce incorrect words.")
            uploaded_video = st.file_uploader(
                "Upload a video of someone signing",
                type=["mp4", "avi", "mov"],
            )

            if uploaded_video is not None:
                with tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False
                ) as tmp:
                    tmp.write(uploaded_video.read())
                    video_path = tmp.name

                try:
                    with st.spinner("Decoding signs..."):
                        feats = extract_video_landmarks(video_path)
                        glosses = sliding_window_predict(
                            model, classes, feats
                        )
                        english = gloss_to_english(glosses)

                    if glosses:
                        st.success(f"**Decoded signs:** {', '.join(glosses)}")
                        st.info(f"**English:** {english or '—'}")
                    else:
                        st.warning("No sign passed the confidence check. Try a clearer/longer clip or record more examples for this PSL word.")

                    if english:
                        config.PLOTS_DIR.mkdir(
                            parents=True,
                            exist_ok=True
                        )
                        audio_path = str(
                            config.PLOTS_DIR / "output_speech.wav"
                        )
                        try:
                            save_audio(english, audio_path)
                            st.audio(audio_path)
                        except Exception as audio_error:
                            st.warning(
                                "Signs and English text were decoded, but speech "
                                "could not be generated on the server. "
                                "Please try again after the app finishes rebuilding. "
                                f"Audio detail: {audio_error}"
                            )

                except Exception as e:
                    st.error(f"Video processing failed: {e}")

                finally:
                    Path(video_path).unlink(missing_ok=True)

        else:
            st.caption(
                "Green landmarks show hand detection. Sign normally; stable signs are converted to text and voice automatically."
            )

            if st.button(
                "🎥 Start Live Camera",
                use_container_width=True
            ):
                script_path = str(
                    Path(__file__).resolve().parent
                    / "predict_live_continuous.py"
                )
                subprocess.Popen(
                    [sys.executable, script_path]
                )
                st.success(
                    "Live camera window launched — check your taskbar."
                )

    with st.expander("🎯 Calibrate an existing PSL sign"):
        st.write("Record camera-specific examples for an existing PSL word that the model confuses.")
        calibration_word = st.selectbox("Choose an existing word", classes)
        sample_count = st.number_input(
            "Number of examples to record", min_value=7, max_value=200,
            value=20, step=5, key="calibration_sample_count",
        )
        calibration_dir = config.DATA_RAW_DIR / "custom_samples" / calibration_word
        calibration_saved = len(list(calibration_dir.glob("sample_*.npy"))) if calibration_dir.exists() else 0
        st.caption(f"Saved examples waiting to add: {calibration_saved}. Press R once per example; after a 2-second countdown, perform the complete sign. Each capture saves automatically.")
        record_col, process_col = st.columns(2)
        with record_col:
            if st.button("📹 Record examples", key="record_existing_word", use_container_width=True):
                launch_word_recorder(calibration_word, int(sample_count))
                st.info("Camera opened. Press R once per example; Q closes the recorder.")
        with process_col:
            if st.button("➕ Add examples & fine-tune", key="process_existing_word",
                         disabled=calibration_saved < 7, use_container_width=True):
                process_word_recordings(calibration_word, 7)

    with st.expander("🆕 Add a new PSL word"):
        st.write("Record examples for a new sign label, add them to the PSL dataset, and train an expanded classifier.")
        raw_new_word = st.text_input("New word label", placeholder="e.g. good_morning", key="new_psl_word")
        new_word = re.sub(r"\s+", "_", raw_new_word.strip().lower())
        valid_new_word = bool(re.fullmatch(r"[a-z0-9]+(?:[_-][a-z0-9]+)*", new_word))
        word_already_exists = valid_new_word and new_word in classes
        new_sample_count = st.number_input(
            "Examples to record (recommended: 40)", min_value=20, max_value=200,
            value=40, step=5, key="new_word_sample_count",
        )
        new_sample_dir = config.DATA_RAW_DIR / "custom_samples" / new_word if valid_new_word else None
        new_saved = len(list(new_sample_dir.glob("sample_*.npy"))) if new_sample_dir and new_sample_dir.exists() else 0
        st.caption(f"Saved examples waiting to add: {new_saved}. Press R once per example; after a 2-second countdown, perform the complete sign. Each capture saves automatically.")
        if word_already_exists:
            st.info("This word is already in the model. Use the existing-word calibration section above.")
        elif raw_new_word and not valid_new_word:
            st.error("Use English letters/numbers and underscores, for example: good_morning")

        record_col, process_col = st.columns(2)
        with record_col:
            if st.button("📹 Record new-word examples", key="record_new_word",
                         disabled=not valid_new_word or word_already_exists,
                         use_container_width=True):
                launch_word_recorder(new_word, int(new_sample_count))
                st.info("Camera opened. Press R once per example; Q closes the recorder.")
        with process_col:
            if st.button("🧠 Add new word & train model", key="process_new_word",
                         disabled=not valid_new_word or word_already_exists or new_saved < 20,
                         use_container_width=True):
                process_word_recordings(new_word, 20)


with col2:
    with st.container(border=True):
        st.header("✍️ Text → Sign")
        st.write(
            "Choose English words with Urdu meanings, then generate "
            "the matching animation."
        )

        input_mode = st.radio(
            "Input method",
            options=["⌨️ Type/Select Words", "🎤 Speak"],
            horizontal=True,
            label_visibility="collapsed",
        )

        words_to_render = []

        if input_mode == "⌨️ Type/Select Words":
            selected_display_labels = st.multiselect(
                "Choose words (in the order you want them signed)",
                options=list(display_to_class.keys()),
                help=(
                    "English words are shown with Urdu meanings. "
                    "Only trained vocabulary can be animated."
                ),
            )

            # Pass only the original English labels to the animation code.
            words_to_render = [
                display_to_class[label]
                for label in selected_display_labels
            ]

        else:
            st.caption(
                "Record a sentence. Only words matching the 68-word English "
                "vocabulary will be signed; unmatched words are skipped."
            )

            audio_value = st.audio_input("Record your voice")

            if audio_value is not None:
                with st.spinner("Transcribing..."):
                    recognizer = sr.Recognizer()

                    try:
                        with sr.AudioFile(audio_value) as source:
                            audio_data = recognizer.record(source)

                        transcript = recognizer.recognize_google(
                            audio_data
                        )

                    except sr.UnknownValueError:
                        transcript = ""
                        st.warning(
                            "Couldn't understand the audio. Try speaking "
                            "clearly, closer to the mic."
                        )

                    except sr.RequestError:
                        transcript = ""
                        st.error(
                            "Speech recognition needs an internet connection. "
                            "Check your connection and try again."
                        )

                if transcript:
                    st.write(f"**Heard:** {transcript}")

                    heard_words = transcript.lower().split()
                    matched = [
                        word for word in heard_words
                        if word in classes
                    ]
                    missing = [
                        word for word in heard_words
                        if word not in classes
                    ]

                    if matched:
                        st.success(
                            f"Matched vocabulary words: {', '.join(matched)}"
                        )

                    if missing:
                        st.warning(
                            f"Not in vocabulary (skipped): {', '.join(missing)}"
                        )

                    words_to_render = matched

        if st.button(
            "🤟 Generate Sign Animation",
            use_container_width=True
        ):
            if not words_to_render:
                st.warning(
                    "No valid vocabulary words to animate yet."
                )

            else:
                with st.spinner("Rendering sign animation..."):
                    gif_path = render_sign_sequence(words_to_render)

                if gif_path:
                    st.image(
                        gif_path,
                        caption=" → ".join(words_to_render),
                        use_container_width=True,
                    )

        with st.expander(
            f"📖 Full vocabulary ({len(classes)} words)"
        ):
            vocabulary_lines = [
                f"{word} — {URDU_MEANINGS.get(word, 'اردو معنی دستیاب نہیں')}"
                for word in classes
            ]
            st.markdown(
                "\n\n".join(
                    f"- {line}" for line in vocabulary_lines
                )
            )

