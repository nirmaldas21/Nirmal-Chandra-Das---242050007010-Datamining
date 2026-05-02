# ============================================================
# Task 2: Naive Bayes Classification - SMS Spam Detection
# Dataset: SMS Spam Collection (UCI / public URL)
# Algorithm: Multinomial Naive Bayes
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_sms_data():
    """Download and load SMS Spam dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    local_zip = "smsspamcollection.zip"
    data_file = "SMSSpamCollection"

    if not os.path.exists(data_file):
        print("Downloading dataset...")
        try:
            urllib.request.urlretrieve(url, local_zip)
            with zipfile.ZipFile(local_zip, 'r') as z:
                z.extractall(".")
            os.remove(local_zip)
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Using fallback demo dataset...")
            return get_demo_data()

    df = pd.read_csv(data_file, sep='\t', header=None, names=['label', 'message'])
    return df


def get_demo_data():
    """Fallback: small hard-coded dataset if download fails."""
    messages = [
        ("ham", "Hey, are you free this weekend?"),
        ("ham", "I'll call you later, take care."),
        ("ham", "Meeting rescheduled to 3pm."),
        ("spam", "Win a FREE iPhone! Click now!"),
        ("spam", "Congratulations! You won $1000. Claim now."),
        ("spam", "URGENT: Your account will be suspended. Verify now."),
        ("ham", "Can you send me the notes?"),
        ("spam", "Get rich quick! Limited offer!"),
        ("ham", "Don't forget mom's birthday tomorrow."),
        ("spam", "FREE entry! Text WIN to 12345!"),
    ] * 50  # Repeat to get more samples
    return pd.DataFrame(messages, columns=['label', 'message'])


#  Load Data 
df = load_sms_data()

print("=" * 50)
print("NAIVE BAYES - SMS SPAM CLASSIFICATION")
print("=" * 50)
print(f"Total Messages : {len(df)}")
print(f"Label Distribution:\n{df['label'].value_counts()}")


#Encode Labels
# Convert text labels to binary: ham=0, spam=1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

print(f"\nVocabulary Size (TF-IDF features): {X.shape[1]}")


# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")


# Train Multinomial Naive Bayes

model = MultinomialNB()
model.fit(X_train, y_train)
print("\nNaive Bayes Model trained successfully!")


#Predict and Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {accuracy * 100:.2f}%")

print("\nClassification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im)
ax.set_title("Confusion Matrix - Spam Detection", fontsize=13)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Ham', 'Spam'])
ax.set_yticklabels(['Ham', 'Spam'])

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontsize=14)

plt.tight_layout()
plt.savefig("nb_confusion_matrix.png", dpi=100)
plt.show()
print("\nConfusion matrix saved as 'nb_confusion_matrix.png'")


# Label Distribution Plot
df['label'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'],
                                 title='Ham vs Spam Count', figsize=(6, 4))
plt.xticks(rotation=0)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("label_distribution.png", dpi=100)
plt.show()
print("Label distribution chart saved as 'label_distribution.png'")
print("\nDone! Task 2 Complete.")