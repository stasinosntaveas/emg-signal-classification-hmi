import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "DB2", "DB2_s1", "DB2_s1", "S1_E1_A1.mat")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

# ========================
# 1. LOAD DATA
# ========================

file_path = DATA_PATH
data = scipy.io.loadmat(file_path)

emg = data['emg']
labels = data['restimulus'].flatten()

print("EMG shape:", emg.shape)
print("Labels shape:", labels.shape)
print("Unique labels:", np.unique(labels))


# ========================
# 2. WINDOWING
# ========================

window_size = 200
step_size = 100

X = []
y = []

for i in range(0, len(emg) - window_size, step_size):
    window = emg[i:i + window_size]
    label_window = labels[i:i + window_size]

    label = np.bincount(label_window).argmax()

    if label != 0:
        X.append(window)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ========================
# 3. FEATURE EXTRACTION
# ========================

def extract_features(window):
    features = []

    for ch in range(window.shape[1]):
        signal = window[:, ch]

        mav = np.mean(np.abs(signal))
        rms = np.sqrt(np.mean(signal ** 2))
        wl = np.sum(np.abs(np.diff(signal)))

        features.extend([mav, rms, wl])

    return features


X_features = np.array([extract_features(w) for w in X])

print("Feature shape:", X_features.shape)


# ========================
# 4. TRAIN / TEST SPLIT
# ========================

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.2,
    random_state=42
)


# ========================
# 5. SCALING
# ========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ========================
# 6. MODELS
# ========================

models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}
best_model_name = None
best_pred = None
best_acc = 0

for name, model in models.items():
    print("\n======================")
    print(name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_pred = y_pred


# ========================
# 7. VISUALISATIONS
# ========================

class_labels = list(range(1, 18))


# ------------------------
# 7.1 Accuracy comparison
# ------------------------

plt.figure(figsize=(6, 4))
plt.bar(list(results.keys()), list(results.values()))
plt.title("Model Accuracy Comparison (EMG Classification)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.show()


# ------------------------
# 7.2 Confusion matrix
# ------------------------

cm = confusion_matrix(y_test, best_pred, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    cmap="Blues",
    annot=False,
    vmin=0,
    vmax=1,
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.title(f"Normalized Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()


# ------------------------
# 7.3 Per-class F1-score
# ------------------------

report = classification_report(y_test, best_pred, output_dict=True)

f1_scores = []
for c in class_labels:
    f1_scores.append(report[str(c)]["f1-score"])

plt.figure(figsize=(10, 4))
plt.bar(class_labels, f1_scores)
plt.title(f"Per-Class F1 Score - {best_model_name}")
plt.xlabel("Gesture Class")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(class_labels)
plt.grid(axis='y', alpha=0.3)
plt.show()


# ========================
# 8. SUMMARY
# ========================

print("\n======================")
print("BEST MODEL:", best_model_name)
print(f"{best_model_name} ACCURACY:", best_acc)



# ========================
# 9. SAVE MODELS
# ========================

import joblib

svm = SVC()
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

svm.fit(X_train, y_train)
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

joblib.dump(svm, os.path.join(RESULTS_DIR, "svm.pkl"))
joblib.dump(knn, os.path.join(RESULTS_DIR, "knn.pkl"))
joblib.dump(rf, os.path.join(RESULTS_DIR, "rf.pkl"))

joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.pkl"))

print("\nAll models saved to /results")

