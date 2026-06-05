import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load Dataset
df = pd.read_csv("heart.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst Five Records:")
print(df.head())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Heart Disease Distribution
plt.figure(figsize=(6,4))
df['target'].value_counts().plot(kind='bar')
plt.title("Heart Disease Distribution")
plt.xlabel("Target")
plt.ylabel("Count")
plt.savefig("heart_disease_distribution.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

# Results Table
results = pd.DataFrame({
    "Algorithm": [
        "Logistic Regression",
        "KNN",
        "Decision Tree"
    ],
    "Accuracy": [
        acc_lr,
        acc_knn,
        acc_dt
    ],
    "F1 Score": [
        f1_lr,
        f1_knn,
        f1_dt
    ]
})

print("\nPerformance Comparison")
print(results)

# Accuracy Graph
plt.figure(figsize=(7,5))
plt.bar(
    results["Algorithm"],
    results["Accuracy"]
)

plt.title("Algorithm Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig("accuracy_comparison.png")
plt.show()

# Best Model
best_model = results.loc[
    results["Accuracy"].idxmax()
]

print("\nBest Model:")
print(best_model)

# Confusion Matrix (Decision Tree)
cm = confusion_matrix(
    y_test,
    y_pred_dt
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Decision Tree Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

print("\nProject Completed Successfully!")
#Due to large numpy arrays, save and load will be done easier
import joblib
joblib.dump(dt, "Heart_Disease_Model.pkl")
model = joblib.load("Heart_Disease_Model.pkl")