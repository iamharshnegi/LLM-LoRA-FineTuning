# Parameter-Efficient Fine-Tuning (PEFT) using LoRA

This repository contains an end-to-end pipeline built on **PyTorch** and **Hugging Face** to fine-tune an open-source transformer model (`DistilBERT`) using **Low-Rank Adaptation (LoRA)** for sequence classification. 

By leveraging PEFT, the pipeline freezes the foundational weights of the model and trains less than **1.1%** of the parameters, drastically cutting down compute costs while maintaining high downstream accuracy.

## 🚀 Performance Metrics & Results
* **Base Model:** `distilbert-base-uncased` (67.6M parameters)
* **Trainable Parameters:** 739,586 (**1.09%** of total parameters)
* **Hardware:** NVIDIA T4 GPU (Google Colab)
* **Training Time:** ~41 seconds (1 Epoch)

### Epoch Evaluation Log

| Epoch | Training Loss | Validation Loss | Accuracy |
| :---: | :-----------: | :-------------: | :------: |
|   1   |   0.587161    |    0.569586     | **81.4%**|

The model successfully converged in a single epoch without overfitting, achieving an **81.4% accuracy** score on completely unseen evaluation data.

## 🛠️ Tech Stack & Frameworks
* **Core Backend:** PyTorch
* **Model Hub & Training:** Hugging Face (Transformers, PEFT, Accelerate, Datasets)
* **Optimization Integration:** TorchAO
* **Dataset:** StanfordNLP IMDB Movie Reviews

## 📦 How to Run the Pipeline
1. Clone the repository:
   ```bash
   git clone https://github.com
   cd LLM-LoRA-FineTuning
   ```
2. Install dependencies:
   ```bash
   pip install transformers datasets peft accelerate evaluate torchao
   ```
