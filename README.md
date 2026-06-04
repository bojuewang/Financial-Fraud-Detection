# 🚨 Elliptic Bitcoin Transaction Classification  
### Semi-Supervised Kernel Logistic Regression with Random Fourier Features

## 📌 Overview
This project focuses on detecting **illicit Bitcoin transactions** using the **Elliptic dataset**, a graph-structured financial dataset widely used for anti-money laundering (AML) research.

Instead of using graph neural networks, this project explores a **feature-based approach**:
- Kernel approximation via **Random Fourier Features (RFF)**
- A **semi-supervised labeling strategy** for handling unknown samples
- A **time-aware training pipeline** to avoid data leakage

The goal is to investigate whether **nonlinear feature models can outperform graph-based methods under certain conditions**.

---

## 🧠 Key Ideas

### Kernel Approximation (RFF)
- Map input features into a higher-dimensional space  
- Train a linear classifier (logistic regression)  
- Achieve nonlinear decision boundaries efficiently  

### Semi-Supervised Learning (Pseudo Labeling)
- Train two auxiliary classifiers:
  - licit vs others  
  - illicit vs others  
- Predict unknown samples and assign pseudo-labels  
- Expand training data  

### Time-Aware Split
- Train: first 70% (earliest time steps)  
- Validation: next 15%  
- Test: last 15%  
- Prevents temporal leakage  

---

## 🏗️ Pipeline

Raw Data  
↓  
Merge Features + Labels  
↓  
Time-based Split  
↓  
Fill Unknown Labels  
↓  
Hyperparameter Tuning (Validation AUC)  
↓  
Retrain Final Model  
↓  
Test Evaluation  
↓  
Save Results  

---

## ⚙️ Model

- Random Fourier Feature Layer  
- Logistic Regression Output  

Formulation:  
- z(x) = sqrt(2/D) * cos(Wx + b)
- output = sigmoid(wᵀz(x))  

---

## 📊 Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC AUC  
- Average Precision  

Composite Score:  
0.35 * ROC AUC + 0.35 * AP + 0.15 * F1 + 0.15 * Recall  

---

## 📈 Outputs

Saved in `elliptic_outputs/`:
- Model checkpoints  
- ROC / PR curves  
- Confusion matrix  
- Training loss  
- Predictions (CSV)  

---

## 🚀 Run

```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn
python your_script.py
