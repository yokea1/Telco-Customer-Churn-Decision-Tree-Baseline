# Telco Customer Churn – Decision Tree Baseline

### 🧠 CSC3600 Intelligent Computing – Lab 9  
**Author:** He Yuke  

---

## 📘 Project Overview

This project implements a **Decision Tree Classifier** to predict customer churn using the **Telco Customer Churn dataset**.  
The goal is to build a baseline model that demonstrates data preprocessing, feature encoding, model training, and evaluation through visual and textual representations.

---

## 🧩 Dataset

The dataset used is:

Copy of Telco-Customer-Churn (lab8).csv

yaml
Copy code

It contains customer-level information such as service plans, contract types, payment methods, and churn status.

---

## ⚙️ Environment Setup

**Requirements:**
- Python 3.8 or above  
- Libraries:
  ```bash
  pip install pandas matplotlib seaborn scikit-learn
🚀 How to Run
Clone this repository:

bash
Copy code
git clone https://github.com/yokea1/Telco-Customer-Churn-Decision-Tree-Baseline.git
cd Telco-Customer-Churn-Decision-Tree-Baseline
Run the Python script:

bash
Copy code
python csc3600lab9.py
The following outputs will be generated:

Classification metrics in terminal (accuracy, precision, recall, F1-score)

Confusion matrix (saved as confusion_matrix.png)

Decision tree visualization (saved as decision_tree.png)

Decision rules (text format) printed in console

📊 Sample Output
yaml
Copy code
Model Accuracy: 0.79

Classification Report:
              precision    recall  f1-score   support
           0       0.85      0.90      0.88       1553
           1       0.63      0.52      0.57        554
    accuracy                           0.81       2107
🌳 Model Visualization
The trained Decision Tree is saved as a high-resolution image:

Copy code
decision_tree.png
It highlights the most important features and their split thresholds.

🧠 Key Concepts Demonstrated
Handling missing and non-numeric data

Encoding categorical variables with one-hot encoding

Splitting data into training and test sets

Building interpretable ML models (Decision Tree)

Evaluating model performance using metrics and confusion matrix

📚 Acknowledgments
This lab exercise is part of the CSC3600 Intelligent Computing course.
Dataset adapted from the Telco Customer Churn dataset (IBM Sample Data).
