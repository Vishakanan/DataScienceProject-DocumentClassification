# A Comparative Study of NLP‑Based Document Classification Techniques on the 20 Newsgroups (Flat vs. Hierarchical)

This repository shares my project comparing **flat** and **hierarchical** approaches for document classification on the **20 Newsgroups** dataset. It includes traditional ML, LSTM, and BERT transfer learning models, plus a small demo app notebook.

> **License:** MIT

---

## What’s Included

- **Flat baseline (non‑hierarchical)**  
  `Document_classfication_with_All_model_Flat_comparison.ipynb` – Compares common flat baselines on 20 Newsgroups with evaluation tables/plots.

- **Hierarchical – Traditional ML**  
  `Traditional_ML_Models_for_hierarchical.ipynb` – Two‑stage pipeline (Level‑1 superclasses → Level‑2 leaf categories) using TF‑IDF + classic ML.

- **Hierarchical – LSTM**  
  `LSTM_Models_for_hierarchical.ipynb` – Tokenization + Embedding + LSTM; trains Level‑1 router and Level‑2 fine classifiers.

- **Hierarchical – BERT / DistilBERT**  
  `TransferLearning_with_Bert_Model_for_hierarchical.ipynb` – Transfer learning with Transformers; two‑stage training and evaluation.

- **Demo app (notebook)**  
  `Document_Classifier_app.ipynb` – Simple interactive notebook UI to classify your own text with a trained model.

- **(Optional files you may see in this repo)**  
  Project report (`.pdf`/`.docx`), presentation (`.pptx`/`.pdf`), generated images (`.png`/`.jpg`), and dataset files if provided.

---

## Problem & Hierarchy (Short)

- **Goal:** Single‑label classification across the 20 Newsgroups categories.  
- **Flat:** Predict one of 20 categories directly.  
- **Hierarchical:** First predict a **superclass** (comp, rec, sci, talk, misc), then predict the **leaf** category within that superclass.

---

##  How to Use

- Click any notebook on GitHub to preview results (tables/plots).  
- If a notebook is large or doesn’t render, download it and open in Jupyter/Colab.  
- The app notebook shows a tiny interface to test a trained model with your own text.

---

## Notes

- Results and plots are inside each notebook.  
- If the dataset is large, it may be shared via Releases or an external link in this repository.  
- Filenames are kept simple so you can upload and manage everything directly in the GitHub web interface.

---

##  Acknowledgments

- 20 Newsgroups dataset (scikit‑learn)  
- Hugging Face Transformers (BERT/DistilBERT)  
- TensorFlow/Keras, PyTorch  
- Gradio (for the demo UI)

---

## License

This project is available under the **MIT License** (see `LICENSE`).
