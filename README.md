### 🚀 **Fine-Tuning BERT for Multi-Label Text Classification**

**Skills:** NLP, Transformer Models, BERT, PyTorch, Hugging Face, Multi-Label Classification  

This project explores **fine-tuning BERT**, a transformer-based **state-of-the-art NLP model**, for **multi-label text classification**. Unlike traditional classification tasks where each text belongs to a single category, multi-label classification requires assigning **multiple independent labels** to a single input.  

To demonstrate this, we use the **Jigsaw Toxic Comment dataset**, which labels online comments into six toxicity categories. The goal is to build a robust **content moderation model** that can detect **toxic, offensive, and harmful content**—a crucial application in **social media monitoring, AI-powered moderation, and online safety**.  

---

## 🎯 **Key Objectives**
✅ **Fine-Tune BERT for Multi-Label Classification**  
✅ **Use Sigmoid Activation & Binary Cross-Entropy Loss** for independent label predictions  
✅ **Leverage GPU Acceleration** with PyTorch for faster training  
✅ **Evaluate Model Performance** using Precision, Recall, F1-score, and AUC-ROC  
✅ **Deploy Model for Real-World Text Moderation Applications**  

---

## 📊 **Dataset Overview: Jigsaw Toxic Comment Dataset**
This dataset consists of **text-based comments** labeled into **six toxicity categories**:  
- **toxic**  
- **severe_toxic**  
- **obscene**  
- **threat**  
- **insult**  
- **identity_hate**  

Unlike **multi-class classification**, where one comment belongs to only one category, **multi-label classification** allows comments to be flagged under **multiple categories simultaneously**.  

---

## 🏗 **Project Pipeline**
1️⃣ **Data Preprocessing** – Cleaning & preparing text for tokenization  
2️⃣ **Tokenization with BERT** – Converting text into transformer-compatible input  
3️⃣ **Dataset Preparation** – Structuring data into PyTorch `Dataset` & `DataLoader`  
4️⃣ **Fine-Tuning BERT** – Training a transformer model on multi-label text data  
5️⃣ **Model Evaluation** – Assessing F1-score, accuracy, precision-recall curves  
6️⃣ **Deployment Considerations** – Saving & optimizing model for real-world usage  

---

## 🔍 **Why Multi-Label Classification is Unique?**
✅ Uses **Sigmoid Activation** (not Softmax) – Each label gets an independent probability  
✅ Requires **Binary Cross-Entropy (BCE) Loss** – Evaluates each label separately  
✅ Uses **Multi-Label Metrics** like **Precision-Recall & F1-Score** instead of Accuracy  

### ✅ **Example Query: Multi-Label Prediction**
```python
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "This comment is awful and extremely offensive!"
tokens = tokenizer(text, return_tensors="pt")

# Forward pass through model
model.eval()
with torch.no_grad():
    predictions = model(**tokens).logits

# Convert logits to probabilities
probabilities = torch.sigmoid(predictions)
labels = (probabilities > 0.5).int()
print(labels)
```
💡 **Output Interpretation:** The model predicts whether the comment is toxic, obscene, or offensive based on thresholding.

---

## 🔥 **BERT Fine-Tuning for Multi-Label Classification**
**Why BERT?**  
🔹 **Context-Aware Representations** – Better understanding of nuanced toxicity  
🔹 **Pre-Trained on Large Corpora** – Requires less labeled data to fine-tune  
🔹 **Handles Multi-Label Tasks Well** – Supports independent label predictions  

✅ **Training Loop Example (PyTorch)**  
```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        inputs, labels = batch
        outputs = model(**inputs).logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
💡 **Key Optimizations:**  
✔ **AdamW Optimizer** – Improves weight updates  
✔ **Learning Rate Scheduling** – Prevents overfitting  

---

## 📊 **Model Evaluation & Metrics**
Since **accuracy is misleading** in multi-label tasks, we use:  
✔ **F1-Score** – Balances precision & recall  
✔ **AUC-ROC** – Measures overall model effectiveness  
✔ **Precision-Recall Curves** – Evaluates how well model distinguishes labels  

✅ **Example: Compute F1-Score**
```python
from sklearn.metrics import f1_score

preds = model.predict(test_dataloader)
f1 = f1_score(y_true, y_pred, average="macro")
print("F1-Score:", f1)
```
💡 **Why F1-Score?** It ensures **both precision (low false positives) & recall (low false negatives)** are optimized.

---

## 🔮 **Future Scope & Enhancements**
🔹 **Fine-Tune DistilBERT for Faster Inference**  
🔹 **Use Ensemble Models for Robustness**  
🔹 **Integrate Explainability (SHAP/LIME) for Transparency**  
🔹 **Deploy Model as an API for Real-Time Toxicity Detection**  

---

## 📌 **References & Tutorials Used**
This project was built using the following tutorials & resources:  
🔹 [Hugging Face BERT Fine-Tuning Guide](https://huggingface.co/transformers/)  
🔹 [Jigsaw Toxic Comment Classification Kaggle Competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/)  
🔹 [PyTorch Transformers Documentation](https://pytorch.org/)  

---

## 🛠 **How to Run This Project**
1️⃣ Clone the repo:  
   ```bash
   git clone https://github.com/shrunalisalian/bert-multi-label-text-classification.git
   ```
2️⃣ Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the Jupyter Notebook:  
   ```bash
   jupyter notebook fine-tuning-bert-multi-label.ipynb
   ```

---

## 📌 **Connect with Me**
- **LinkedIn:** [Shrunali Salian](https://www.linkedin.com/in/shrunali-salian/)  
- **Portfolio:** [https://portfolio-shrunali-suresh-salians-projects.vercel.app/](#)  
- **Email:** [Your Email](#)  
