
"""
Two-way PSL communication app.

Left panel:  Sign -> Text / Voice
Right panel: Text -> Sign

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


st.set_page_config(
    page_title="PSL Two-Way Communication",
    layout="wide",
    page_icon="🤟",
)

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
            st.caption("Video is split at short no-hand gaps; pause briefly between signs when the clip contains multiple signs.")
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

                    st.success(
                        f"**Decoded signs:** "
                        f"{', '.join(glosses) if glosses else '(none detected)'}"
                    )
                    st.info(f"**English:** {english or '—'}")

                    if english:
                        config.PLOTS_DIR.mkdir(
                            parents=True,
                            exist_ok=True
                        )
                        audio_path = str(
                            config.PLOTS_DIR / "output_speech.wav"
                        )
                        save_audio(english, audio_path)
                        st.audio(audio_path)

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
        st.write("Record more examples for a word the current model misses. New vocabulary can be added later.")
        calibration_word = st.selectbox("Choose an existing word", classes)
        sample_count = st.number_input(
            "Number of samples to record",
            min_value=7,
            max_value=200,
            value=20,
            step=5,
        )

        if st.button("📹 Record calibration samples"):
            recorder = Path(__file__).resolve().parent / "record_word.py"
            subprocess.Popen([
                sys.executable,
                str(recorder),
                "--word", calibration_word,
                "--samples", str(int(sample_count)),
            ])
            st.success("Camera window khul gaya. Green points detection dikhayenge; R se sample start/stop, Q se finish karein.")

        st.caption("Recording ke baad terminal mein yeh commands chalayein. Training complete ho to Streamlit restart karein.")
        st.code(
            "python src/add_recorded_words.py\n"
            "python src/train.py --resume\n"
            "python src/build_templates.py\n"
            "python src/evaluate.py",
            language="powershell",
        )


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
