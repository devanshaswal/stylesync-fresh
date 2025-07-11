This is the final-year dissertation project that I completed in my final year at Brunel University London and got an A grade on.
# StyleSync 👗📊

**A Multimodal Fashion Recommendation System using ResNet and Vision Transformer**

This repository contains the implementation of **StyleSync**, a final-year dissertation project submitted to **Brunel University London** by **Devansh Aswal** (BSc Computer Science with AI). The system performs intelligent fashion recommendations using both **ResNet-50** and **Vision Transformer (ViT)** models, trained on the **DeepFashion-MultiModal** dataset.

---

## 🔍 Project Summary

StyleSync predicts:
- Fashion **category** (e.g. dress, shirt),
- **Category type** (upper-body, lower-body, full-body),
- Multi-label **attributes** (e.g. colour, neckline, fabric)

### Key Features:
- 🔁 RGB image + **landmark heatmap fusion**
- ⚖️ **Focal loss** + **WeightedRandomSampler** for class imbalance
- 🧠 ViT with **grouped attribute heads** and **cross-modal attention**
- 🎯 Multi-task training with **deep supervision**

---

## 📁 Dataset

- **DeepFashion-MultiModal**
- Preprocessed using bounding boxes and landmark annotations
- 50 selected attributes (from original 1000)

---

## 🛠️ Technologies

- Python, PyTorch
- Google Colab with A100 GPU (40GB)
- Mixed precision & gradient checkpointing

---

## 📊 Test Set Results

| Task                | Accuracy |
|---------------------|----------|
| Category Prediction | 11.2%    |
| Category Type       | 49.2%    |

F1 scores were used to evaluate multi-label attributes across both models.

---

## 📄 Citation

**Aswal, D. (2025)**. *StyleSync: A Multimodal Approach to Intelligent Fashion Recommendation System*. Brunel University London.

---

## 📬 Contact

Email: devansh.aswal@gmail.com  
LinkedIn: [linkedin.com/in/devansh-aswal](https://www.linkedin.com/in/devansh-aswal/)
