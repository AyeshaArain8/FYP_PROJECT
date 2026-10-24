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

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

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
    ├── inference_continuous.py  # sliding-window continuous sentence decoding
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

3. **Try the live webcam demo (single sign):**
   ```bash
   python src/predict_live.py
   ```

4. **Try continuous sentence decoding on a recorded video:**
   ```bash
   python src/inference_continuous.py path/to/video.mp4
   ```

5. **Run the full two-way app:**
   ```bash
   streamlit run src/app.py
   ```

## Important things to know before you defend this

- **This is isolated-sign training data, not continuous-sentence data.**
  Every sample in the dataset is one gesture performed alone. The
  "continuous" behavior comes from `inference_continuous.py` running a
  sliding window over a longer video and stitching together a sequence of
  independently-recognized signs — not from the model itself understanding
  sentence structure. Be ready to explain this distinction clearly; it's
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

- **Hand left/right ordering in `landmark_utils.py` is an assumption**
  (left hand first, right hand second) since the original dataset's exact
  convention isn't documented. If live predictions look consistently wrong
  on asymmetric one-handed signs, that ordering is the first thing to check
  and flip.

- **7 classes were automatically dropped** during preprocessing: `test2`,
  `test_word`, `s` (junk/incomplete recordings), and `bird`, `come`,
  `give`, `want` (too few real samples to train on reliably, <40 each).
  You can lower `MIN_SAMPLES_PER_CLASS` in `config.py` and re-run
  `src/preprocess.py` if you'd rather keep the low-sample ones and accept
  noisier accuracy on them.

## Dataset citation

Landmark data derived from the "PakistanSignLanguageDatasetV2" community
dataset. Cite its source in your report/thesis references section.
