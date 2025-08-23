# DataScienceProject-DocumentClassification
# A Comparative Study of NLP-Based Document Classification Techniques on the 20 Newsgroups with Hierarchical Modeling

This repository contains a complete, step‑by‑step study comparing **flat** vs **hierarchical** document‑classification approaches on the **20 Newsgroups** dataset. It includes traditional machine learning, LSTM, and BERT‑based transfer learning, plus a simple interactive demo app notebook.

> **License:** MIT (already included in this repo)

---

## Table of Contents

- [Overview](#overview)
- [Files in this Repository](#files-in-this-repository)
- [Environment Setup](#environment-setup)
  - [Pip (recommended)](#pip-recommended)
  - [Conda (optional)](#conda-optional)
  - [NLTK Resources](#nltk-resources)
- [Dataset Options](#dataset-options)
- [How to Run the Notebooks](#how-to-run-the-notebooks)
  - [1) Flat Baselines](#1-flat-baselines)
  - [2) Hierarchical — Traditional ML](#2-hierarchical--traditional-ml)
  - [3) Hierarchical — LSTM](#3-hierarchical--lstm)
  - [4) Hierarchical — BERT / DistilBERT](#4-hierarchical--bert--distilbert)
  - [5) Demo App Notebook](#5-demo-app-notebook)
- [Evaluation & Results Guidance](#evaluation--results-guidance)
- [Reproducibility Tips](#reproducibility-tips)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [How to Cite](#how-to-cite)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

- **Goal:** Compare flat (single stage) document classification against hierarchical (two‑stage) classification on **20 Newsgroups**.
- **Flat:** Directly predict one of the 20 categories.
- **Hierarchical:** First predict a **Level‑1 superclass**, then predict the **Level‑2 leaf** inside that superclass.
- **Why hierarchical?** By narrowing the decision space, Level‑2 classifiers can specialize within a superclass, often improving macro‑F1 and interpretability.

**Hierarchy used (5 superclasses → 20 leaves):**

- **comp:** `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`  
- **rec:** `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey`  
- **sci:** `sci.crypt`, `sci.electronics`, `sci.med`, `sci.space`  
- **talk:** `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`, `talk.religion.misc`  
- **misc:** `alt.atheism`, `soc.religion.christian`, `misc.forsale`

---

## Files in this Repository

Core notebooks (kept at the **top level**, no extra folders):

- **Flat baseline notebook**  
  [`Document_classfication_with_All_model_Flat_comparison.ipynb`](Document_classfication_with_All_model_Flat_comparison.ipynb)  
  Compares flat baselines with classic ML models and evaluation plots/tables.

- **Hierarchical — Traditional ML**  
  [`Traditional_ML_Models_for_hierarchical.ipynb`](Traditional_ML_Models_for_hierarchical.ipynb)  
  Two‑stage pipeline using TF‑IDF + linear/NB classifiers.

- **Hierarchical — LSTM**  
  [`LSTM_Models_for_hierarchical.ipynb`](LSTM_Models_for_hierarchical.ipynb)  
  Tokenization → Embedding + LSTM for Level‑1 router and Level‑2 fine classifiers.

- **Hierarchical — BERT/DistilBERT**  
  [`TransferLearning_with_Bert_Model_for_hierarchical.ipynb`](TransferLearning_with_Bert_Model_for_hierarchical.ipynb)  
  Hugging Face Transformers fine‑tuning for Level‑1 and Level‑2.

- **Interactive demo app (notebook)**  
  [`Document_Classifier_app.ipynb`](Document_Classifier_app.ipynb)  
  Gradio UI to try a trained model on custom text.

Optional artifacts you might include:

- **Project report** (e.g., `Project_Report.pdf` or `.docx`)  
- **Presentation** (e.g., `Presentation.pptx` or `.pdf`)  
- **Images/Figures** (e.g., confusion matrices, charts; you can keep them at top level or in an `images/` folder)  
- **Dataset files** (if you choose to store them here; see the note on GitHub size limits below)

> **Note:** GitHub blocks single files larger than **100 MB** in normal commits. See **[Dataset Options](#dataset-options)** if yours is bigger.

---

## Environment Setup

### Pip (recommended)

Works on CPU; GPU is recommended for LSTM/BERT training.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip

# Core scientific stack
pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm wordcloud

# NLP helpers
pip install nltk

# Deep learning toolkits
pip install tensorflow              # for LSTM notebook
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or your CUDA wheel
pip install transformers            # for BERT notebook

# App / UI
pip install gradio streamlit

# Optional: notebook runtime
pip install jupyterlab
```

### Conda (optional)

```bash
conda create -n 20ng-hier python=3.10 -y
conda activate 20ng-hier
pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm wordcloud nltk tensorflow transformers gradio streamlit jupyterlab
# Choose the correct PyTorch build for your system (CPU or CUDA) from pytorch.org
```

### NLTK Resources

First run only (or if you see NLTK download errors):

```bash
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
```

---

## Dataset Options

You can use **either** approach:

1) **Auto‑download (easiest):**  
   Most notebooks call `sklearn.datasets.fetch_20newsgroups(...)` to download/cache the dataset automatically on first run.

2) **Local dataset file(s) (manual):**  
   - If you uploaded dataset files to the repo, update the data‑loading cells to point to your path/filenames.  
   - **Big files (>100 MB each)** cannot be uploaded via normal commits in the web UI. Two options:  
     - **Releases (recommended):** Go to **Releases → Draft a new release → Attach your dataset files** as assets. Link them from this README.  
     - **External host:** Google Drive/OneDrive/Kaggle/Hugging Face Datasets and link here.

---

## How to Run the Notebooks

Open the `.ipynb` in your preferred environment (Jupyter, VS Code, or Colab) and run cells in order. If using Colab, upload the notebook and run directly (GPU runtime recommended for LSTM/BERT).

### 1) Flat Baselines
**Notebook:** [`Document_classfication_with_All_model_Flat_comparison.ipynb`](Document_classfication_with_All_model_Flat_comparison.ipynb)

- Loads 20NG, cleans text, builds **TF‑IDF** features (uni/bi‑grams if configured).  
- Trains standard **flat** baselines: Multinomial Naive Bayes, Logistic Regression, Linear SVM (and variants if present).  
- Outputs **classification reports** (per‑class + macro‑F1), **confusion matrices**, and **comparison tables**.

### 2) Hierarchical — Traditional ML
**Notebook:** [`Traditional_ML_Models_for_hierarchical.ipynb`](Traditional_ML_Models_for_hierarchical.ipynb)

- Two‑stage classifier: **Level‑1 router** (superclass) → **Level‑2 fine classifier** per superclass.  
- **TF‑IDF** features; linear models/NB for efficiency and interpretability.  
- Handles single‑subcategory groups gracefully.  
- Reports per‑level and overall metrics; includes confusion matrices and per‑group reports.

### 3) Hierarchical — LSTM
**Notebook:** [`LSTM_Models_for_hierarchical.ipynb`](LSTM_Models_for_hierarchical.ipynb)

- Text tokenization, sequence padding, **Embedding + LSTM** models.  
- Trains Level‑1 LSTM router and Level‑2 LSTM fine classifiers.  
- Shows training curves and evaluates accuracy + macro‑F1; plots confusion matrices.

### 4) Hierarchical — BERT / DistilBERT
**Notebook:** [`TransferLearning_with_Bert_Model_for_hierarchical.ipynb`](TransferLearning_with_Bert_Model_for_hierarchical.ipynb)

- Uses **Hugging Face Transformers** (tokenizer + model).  
- Typical config fields: learning rate, batch size, epochs, warmup/weight decay, early stopping.  
- Trains Level‑1 and Level‑2; saves fine‑tuned model and label mappings (if enabled).  
- Reports accuracy + macro‑F1; prints classification reports + confusion matrices; includes per‑group summaries.

### 5) Demo App Notebook
**Notebook:** [`Document_Classifier_app.ipynb`](Document_Classifier_app.ipynb)

- **Gradio** app: paste/enter text → get predicted label + confidence.  
- Point the notebook to your saved model directory (BERT or another trained checkpoint) as described inside the notebook.  
- On run, Gradio prints a local (and sometimes public) URL to open the app in the browser.

---

## Evaluation & Results Guidance

- **Primary metrics:** accuracy and **macro precision/recall/F1** (macro‑F1 is crucial for class imbalance).  
- **Classification reports:** inspect per‑class performance to find weak classes.  
- **Confusion matrices:** spot systematic confusions; compare flat vs hierarchical.  
- **What to expect:** BERT generally outperforms classic ML/LSTM on 20NG after tuning; hierarchical routing can further help by reducing the search space at Level‑2.

> Exact numbers vary by random seed, preprocessing choices (e.g., stopword lists), model hyperparameters, and hardware (CPU/GPU).

---

## Reproducibility Tips

- Fix seeds in NumPy/TensorFlow/PyTorch (the notebooks include seed cells where relevant).  
- Keep scikit‑learn’s train/test split consistent to compare fairly.  
- Capture versions when you finalize results:
  ```bash
  pip freeze > requirements.lock.txt
  ```
- For GPU: ensure your CUDA/cuDNN versions match your PyTorch/TensorFlow wheels.

---

## Troubleshooting

- **`NameError: y_test is not defined`** → Run the cell that creates `X_train, X_test, y_train, y_test` before evaluation cells.  
- **Confusion matrix shape mismatch** → Pass **predicted labels** (not probabilities) and ensure `labels=range(n_classes)` or correct class list.  
- **NLTK resource errors** → Run the NLTK downloader command in [NLTK Resources](#nltk-resources).  
- **Out‑of‑memory on BERT/LSTM** → Reduce `batch_size` and `max_length`; use DistilBERT; prefer GPU.  
- **Gradio app not reachable** → In local runs, open the `http://127.0.0.1:...` URL shown in the output. In Colab, use the public share link printed by Gradio.

---

## FAQ

**Q: Can I keep everything at the top level?**  
A: Yes. This README is written for a flat structure. You can still create simple folders like `images/` for clarity if you want.

**Q: My dataset is too large to commit. What should I do?**  
A: Use **GitHub Releases** and attach the dataset as a release asset, or host externally (Drive/Kaggle/HF Datasets) and link it here.

**Q: Do I need GPU?**  
A: No for classic ML; recommended for LSTM/BERT training to save time.

---

## How to Cite

If you use this repository or its results, please cite:

```bibtex
@software{20ng_hier_doc_classification,
  title  = {A Comparative Study of NLP-Based Document Classification Techniques on the 20 Newsgroups with Hierarchical Modeling},
  author = {Your Name},
  year   = {2025},
  note   = {MIT License},
  url    = {https://github.com/<your-username>/<your-repo>}
}
```

---

## Acknowledgments

- 20 Newsgroups dataset via **scikit‑learn**
- **Hugging Face Transformers** (BERT/DistilBERT)
- **TensorFlow/Keras** and **PyTorch**
- **Gradio** / **Streamlit** for quick demos

---

## License

This project is released under the **MIT License**. See the `LICENSE` file in this repository.
