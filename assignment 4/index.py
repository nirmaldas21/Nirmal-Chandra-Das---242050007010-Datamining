# Task 4: Cancer Dataset Classification + Confusion Matrix
# Dataset: sklearn Breast Cancer dataset
# Algorithm: Logistic Regression


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')


# Step 1: Load Dataset
cancer = datasets.load_breast_cancer()
X = cancer.data         # 30 features
y = cancer.target       # 0 = malignant, 1 = benign

print("=" * 55)
print("CANCER CLASSIFICATION - LOGISTIC REGRESSION")
print("=" * 55)
print(f"Total Samples   : {X.shape[0]}")
print(f"Total Features  : {X.shape[1]}")
print(f"Classes         : {list(cancer.target_names)}")
print(f"\nClass Distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  {cancer.target_names[cls]:12s} : {cnt} samples")


# Step 2: Show Sample Feature Names
print(f"\nSome Features: {list(cancer.feature_names[:6])} ...")


# Step 3: Feature Scaling
# Logistic Regression converges better with scaled features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")


# Step 5: Train Logistic Regression
# max_iter increased to ensure convergence
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)
print("\nLogistic Regression Model trained successfully!")


# Step 6: Predict
y_pred = model.predict(X_test)


#Step 7: Accuracy and Report
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {accuracy * 100:.2f}%")

print("\nClassification Report:")
print("-" * 55)
print(classification_report(y_test, y_pred,
                             target_names=cancer.target_names))


# Step 8: Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (raw):")
print(cm)
print()

# Detailed label extraction
TN = cm[0][0]  # Actual benign, Predicted benign
FP = cm[0][1]  # Actual benign, Predicted malignant
FN = cm[1][0]  # Actual malignant, Predicted benign  <-- most dangerous
TP = cm[1][1]  # Actual malignant, Predicted malignant

print("Confusion Matrix Breakdown:")
print(f"  True Negatives  (Benign   → Benign)   : {TN}  ✅ Correct")
print(f"  False Positives (Benign   → Malignant): {FP}  ⚠️  Over-diagnosed")
print(f"  False Negatives (Malignant→ Benign)   : {FN}  ❌ DANGEROUS - Missed cancer!")
print(f"  True Positives  (Malignant→ Malignant): {TP}  ✅ Correct")

# Derived metrics
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"\nManually Computed Metrics (for Malignant class):")
print(f"  Precision = {precision:.4f}  (of all predicted malignant, how many are really?)")
print(f"  Recall    = {recall:.4f}  (of all actual malignant, how many did we catch?)")
print(f"  F1 Score  = {f1:.4f}")


#Step 9: Plot Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Confusion Matrix Heatmap
ax = axes[0]
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_title("Confusion Matrix\nBreast Cancer Classification", fontsize=12)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
labels = ['Malignant', 'Benign']
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

cell_labels = [
    [f"TN\n{TN}", f"FP\n{FP}"],
    [f"FN\n{FN}", f"TP\n{TP}"]
]
for i in range(2):
    for j in range(2):
        ax.text(j, i, cell_labels[i][j],
                ha='center', va='center',
                color='white' if cm[i][j] > cm.max() / 2 else 'black',
                fontsize=12, fontweight='bold')

# Plot 2: ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

ax2 = axes[1]
ax2.plot(fpr, tpr, color='darkorange', lw=2,
         label=f"ROC Curve (AUC = {roc_auc:.2f})")
ax2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
ax2.set_title("ROC Curve - Cancer Classification", fontsize=12)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate (Recall)")
ax2.legend(loc="lower right")
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("cancer_results.png", dpi=100)
plt.show()
print("\nResults plot saved as 'cancer_results.png'")
print("\nDone! Task 4 Complete.")