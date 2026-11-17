# PSL Two-Way Communication System

A continuous Pakistan Sign Language recognition and two-way translation
system: Sign → Text → Voice, and Text → Sign, built on top of the
"PakistanSignLanguageDatasetV2" landmark dataset (72 words/phrases, 3
recording devices).

## What's already done for you

- ✅ Raw dataset already **preprocessed** into `data/processed/` (train/val/test
  splits, junk classes filtered out, 68 usable classes, 6,834 samples)
- ✅ Sign **templates** built (`data/processed/templates.npz`) for the
  Text → Sign direction
- ✅ Full pipeline code written and **smoke-tested end-to-end** (ran 3 epochs
  to confirm everything loads and trains correctly — reached 67% validation
  accuracy in just 3 epochs, which is a good sign)
- ✅ Model architecture, training script, evaluation script, continuous
  sentence inference, gloss-to-English, text-to-speech, text-to-sign
  animation, and a Streamlit app all wired together

## What you need to do

**Run the actual full training** (the smoke test only ran 3 epochs to verify
correctness — real training needs the full run, see below), then evaluate
and demo it.

## Setup

Use Python 3.10. If this project already has a working `.venv`, activate that
environment and keep using it; there is no need to recreate it. MediaPipe is
pinned to the version used by this project because newer releases removed the
`solutions.hands` API used by the live extractor.

```bash
py -3.10 -m venv .venv          # only for a fresh setup
.venv\Scripts\Activate.ps1     # Windows PowerShell
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

For a project that already runs, do not run the environment creation command.

## Project structure

```
PSL_Communication_System/
├── config.py                    # all paths + hyperparameters live here
├── requirements.txt
├── data/
│   ├── raw/                     # (optional) put archive__4_.zip here if you
│   │                             #  want to re-run preprocessing yourself
│   └── processed/                # ALREADY GENERATED for you
│       ├── X_train.npy / y_train.npy
│       ├── X_val.npy   / y_val.npy
│       ├── X_test.npy  / y_test.npy
│       ├── label_map.json       # class index <-> word name
│       └── templates.npz        # one reference landmark sequence per word
├── outputs/
│   ├── checkpoints/             # trained model saves here
│   └── plots/                   # training curves, confusion matrix, demo outputs
└── src/
    ├── preprocess.py            # (optional) re-run if you change filtering rules
    ├── landmark_utils.py        # MediaPipe hand landmark extractor (matches dataset format)
    ├── model.py                 # BiLSTM classifier architecture
    ├── train.py                 # <-- RUN THIS to train
    ├── evaluate.py               # <-- RUN THIS after training, to see accuracy/confusion matrix
    ├── build_templates.py       # already run once; re-run only if you re-preprocess
    ├── predict_live.py          # webcam demo, single sign at a time
    ├── inference_continuous.py  # classify hand-detected sign segments in videos
    ├── gloss_to_english.py      # rule-based gloss -> English sentence cleanup
    ├── tts.py                   # text-to-speech (pyttsx3, offline)
    ├── text_to_sign.py          # Text -> Sign, renders an animated stick figure
    └── app.py                   # <-- Streamlit two-way app
```

## Run order

1. **Train the model:**
   ```bash
   python src/train.py
   ```
   Trains the BiLSTM classifier on the 68-class vocabulary. Saves the best
   checkpoint (by validation accuracy) to `outputs/checkpoints/psl_model.h5`
   and a loss/accuracy curve plot to `outputs/plots/training_curves.png`.
   Adjust `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE` in `config.py` if needed.

2. **Evaluate on the held-out test set:**
   ```bash
   python src/evaluate.py
   ```
   Prints per-class precision/recall/F1 and saves a confusion matrix image
   to `outputs/plots/confusion_matrix.png`. This is what tells you which
   signs the model actually confuses — important for your thesis results
   section.

3. **Try automatic live webcam sign recognition:**
   ```bash
   python src/predict_live_continuous.py
   ```

   The live camera displays green landmarks and automatically outputs a stable
   sign as text and speech. `c` clears the sentence; `q` closes the camera.

4. **Try continuous sentence decoding on a recorded video:**
   ```bash
   python src/inference_continuous.py path/to/video.mp4
   ```

5. **Run the full two-way app:**
   ```bash
   streamlit run src/app.py
   ```

## Calibrate an existing word or add a new PSL word

In the app, open **Calibrate an existing PSL sign** to add examples to a word
already in the model, or **Add a new PSL word** to extend the vocabulary.
Enter a label such as `good_morning`, then start recording. In the camera
window, press **R once per example**. After the countdown, perform the full
sign; the sample saves automatically. The normal live sign-to-text camera does
not require R. Green landmarks show whether the recording camera sees the hand.

Record at least 7 examples to calibrate an existing class. A new class needs at
least 20; 40 varied examples are recommended. After recording, click the
matching **Add ... & train** button in the app. It integrates the samples,
fine-tunes or expands the model, rebuilds sign templates, evaluates, and reloads
the app. A safety snapshot of model and dataset files is created before each
training run.

This keeps the current vocabulary and fine-tunes the existing classifier.
Restart Streamlit after training to load the updated checkpoint.

## Important things to know before you defend this

- **This is isolated-sign training data, not continuous-sentence data.**
  Every sample in the dataset is one gesture performed alone. The
  video behavior comes from `inference_continuous.py` splitting clips at brief
  hand-detection gaps and classifying each segment — not from the model itself
  understanding sentence structure. Be ready to explain this distinction; it's
  the honest and correct way to describe what you built.

- **`gloss_to_english.py` is a simple rule-based placeholder**, not a
  trained language model. It's intentionally kept as its own module so you
  can swap in a small seq2seq model later without touching anything else.

- **`text_to_sign.py` renders an animated stick figure**, not real video
  footage, because there's no filmed video library for these signs — only
  landmark data. This is a legitimate technical choice worth stating
  explicitly in your report (it reuses the same landmark data the model
  trained on, rather than requiring a second, separately-collected video
  asset library).

- **Live hand slots follow the processed training data:** one detected hand
  goes in slot two; with two detected hands, the hand farther right in image
  coordinates goes in slot one. MediaPipe's left/right label is not used for
  slot assignment because it can change with camera mirroring.

- **7 classes were automatically dropped** during preprocessing: `test2`,
  `test_word`, `s` (junk/incomplete recordings), and `bird`, `come`,
  `give`, `want` (too few real samples to train on reliably, <40 each).
  You can lower `MIN_SAMPLES_PER_CLASS` in `config.py` and re-run
  `src/preprocess.py` if you'd rather keep the low-sample ones and accept
  noisier accuracy on them.

## Dataset citation

Landmark data derived from the "PakistanSignLanguageDatasetV2" community
dataset. Cite its source in your report/thesis references section.
