#Task 1: Digit Recognition using Supervised Learning (SVM)
# Dataset: sklearn digits dataset
# Algorithm: Support Vector Machine (SVM)

# --- Import Libraries ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# Loading thy Dataset
digits = datasets.load_digits()
X = digits.data       # Feature matrix: (1797, 64)
y = digits.target     # Labels: 0 to 9

print("=" * 50)
print("DIGIT RECOGNITION - SVM CLASSIFIER")
print("=" * 50)
print(f"Total Samples   : {X.shape[0]}")
print(f"Features (pixels): {X.shape[1]}")
print(f"Classes         : {np.unique(y)}")


#Visualizing Sample Digits
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Sample Digits from Dataset", fontsize=14)
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("sample_digits.png", dpi=100)
plt.show()
print("\nSample digit images saved as 'sample_digits.png'")


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")


#Train SVM Classifier
model = SVC(kernel='rbf', gamma=0.001, C=10, random_state=42)
model.fit(X_train, y_train)
print("\nSVM Model trained successfully!")


# Predict
y_pred = model.predict(X_test)


#  Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print("\nClassification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred))


# Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im)
ax.set_title("Confusion Matrix - Digit Recognition", fontsize=13)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
tick_marks = np.arange(10)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(range(10))
ax.set_yticklabels(range(10))

# Annotate each cell with its count
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.show()
print("\nConfusion matrix saved as 'confusion_matrix.png'")
print("\nDone! Task 1 Complete.")