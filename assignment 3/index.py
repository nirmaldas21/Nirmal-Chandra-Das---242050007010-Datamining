# Task 3: K-Nearest Neighbors (KNN) on IRIS Dataset
# Dataset: sklearn IRIS dataset
# Algorithm: KNN with comparison of K values

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# Step 1: Load IRIS Dataset
iris = datasets.load_iris()
X = iris.data          # 4 features
y = iris.target        # 3 classes: 0, 1, 2

# Create a DataFrame for easy viewing
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = [iris.target_names[i] for i in y]

print("=" * 50)
print("KNN CLASSIFIER - IRIS DATASET")
print("=" * 50)
print(f"Total Samples : {X.shape[0]}")
print(f"Features      : {iris.feature_names}")
print(f"Classes       : {list(iris.target_names)}")
print(f"\nFirst 5 rows of dataset:")
print(df.head())
print(f"\nClass Distribution:\n{df['Species'].value_counts()}")


# Step 2: Feature Scaling
# KNN uses distance, so scaling is important
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 3: Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")


# Step 4: Try Different K Values
# Experiment with K from 1 to 15 and record accuracy
k_values = range(1, 16)
accuracies = []

print("\nAccuracy for different values of K:")
print("-" * 30)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    print(f"  K = {k:2d}  -->  Accuracy = {acc * 100:.2f}%")


# Step 5: Plot K vs Accuracy
plt.figure(figsize=(9, 5))
plt.plot(k_values, [a * 100 for a in accuracies],
         marker='o', color='steelblue', linewidth=2, markersize=7)
plt.title("KNN: K Value vs Test Accuracy (IRIS Dataset)", fontsize=13)
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, 16))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("k_vs_accuracy.png", dpi=100)
plt.show()
print("\nK vs Accuracy graph saved as 'k_vs_accuracy.png'")


# Step 6: Train Final Model with Best K 
best_k = k_values[np.argmax(accuracies)]
print(f"\nBest K Value : {best_k}")
print(f"Best Accuracy: {max(accuracies) * 100:.2f}%")

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)


# Step 7: Evaluation
print("\nClassification Report (Best K):")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# Step 8: Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Greens')
plt.colorbar(im)
ax.set_title(f"Confusion Matrix - KNN (K={best_k})", fontsize=13)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(iris.target_names, rotation=15)
ax.set_yticklabels(iris.target_names)

for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontsize=14)

plt.tight_layout()
plt.savefig("knn_confusion_matrix.png", dpi=100)
plt.show()
print("\nConfusion matrix saved as 'knn_confusion_matrix.png'")
print("\nDone! Task 3 Complete.")