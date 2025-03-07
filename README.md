# Food Serving System
This repository contains the code for a technical interview I participated in.

## Table of Contents
- [Synthetic Dataset](#synthetic-dataset)
- [Retrieval Augmented Generation](#retrieval-augmented-generation)
- [Reformat Data for Instruction Tuning](#reformat-data-for-instruction-tuning)

---

## Synthetic Dataset
### Goals
- Generate a 50-page dataset in Bahasa Indonesia with random topics in PDF format.
- Generate an SQL database.

The dataset is stored in **Part_II**.

---

## Retrieval Augmented Generation
### Goal
Develop a Retrieval-Augmented Generation (RAG) system that can answer questions based on the synthetic dataset.

### Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run Part_II/app.py
```

### Live Demo
Check out the live demo: [Food Serving System](https://foodservingsystem.streamlit.app/)

---

## Reformat Data for Instruction Tuning
### Goal
Reformat the dataset from **Part_I** to fine-tune four different foundational models:
- **Llama**
- **Gemma**
- **Mistral**
- **Phi**

The reformatted dataset is available in **Part_III**.

---
