# Lab 9
# CSC3600 Intelligent Computing
# Author: He yuke
#Decision Tree for Customer Churn Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
df = pd.read_csv("Copy of Telco-Customer-Churn (lab8).csv")

# 2. Basic cleaning
df.dropna(inplace=True)
df.drop(columns=['customerID'], inplace=True)

# Convert TotalCharges to numeric (handle possible non-numeric values)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert target label to binary
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# One-hot encode categorical features
X = pd.get_dummies(df.drop('Churn', axis=1))
y = df['Churn']

# 3. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 7. Print tree rules (text format)
print("\nDecision Rules (text format):\n")
print(export_text(clf, feature_names=list(X.columns)))

# 8. Visualize the decision tree - High resolution
plt.figure(figsize=(40, 24))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization (High Resolution)", fontsize=16)
plt.savefig("decision_tree.png", dpi=300, bbox_inches='tight')
plt.show()
