import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========================
# PATHS
# ========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "DB2")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

# ========================
# HELPER FUNCTIONS
# ========================

def get_subject_path(subject, exercise=1):
    return os.path.join(
        DATA_DIR,
        f"DB2_s{subject}",
        f"DB2_s{subject}",
        f"S{subject}_E{exercise}_A1.mat"
    )

def extract_features(window):
    features = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]

        mav = np.mean(np.abs(signal))
        rms = np.sqrt(np.mean(signal ** 2))
        wl = np.sum(np.abs(np.diff(signal)))

        features.extend([mav, rms, wl])

    return features

def process_subject(file_path):
    data = scipy.io.loadmat(file_path)

    emg = data['emg']
    labels = data['restimulus'].flatten()

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

    X_features = np.array([extract_features(w) for w in X])

    return X_features, y

# ========================
# SUBJECT SPLIT
# ========================

train_subjects = list(range(1,6))
test_subject = 40

X_train_list, y_train_list = [], []

print("Processing training subjects...")
for s in train_subjects:
    print(f"Subject {s}")
    X_s, y_s = process_subject(get_subject_path(s))
    X_train_list.append(X_s)
    y_train_list.append(y_s)

print(f"\nProcessing test subject {test_subject}...")
X_test, y_test = process_subject(get_subject_path(test_subject))

# Concatenate training data
X_train = np.vstack(X_train_list)
y_train = np.hstack(y_train_list)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ========================
# SCALING
# ========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========================
# MODELS
# ========================

models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
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
# VISUALIZATIONS
# ========================

class_labels = list(range(1, 18))

# Accuracy comparison
plt.figure(figsize=(6, 4))
plt.bar(list(results.keys()), list(results.values()))
plt.title("Model Accuracy Comparison (Cross-Subject EMG)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Confusion matrix
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

# Per-class F1-score
report = classification_report(y_test, best_pred, output_dict=True)

f1_scores = []
for c in class_labels:
    if str(c) in report:
        f1_scores.append(report[str(c)]["f1-score"])
    else:
        f1_scores.append(0)

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
# SUMMARY
# ========================

print("\n======================")
print("BEST MODEL:", best_model_name)
print(f"{best_model_name} ACCURACY:", best_acc)

# ========================
# SAVE MODELS
# ========================

import joblib

os.makedirs(RESULTS_DIR, exist_ok=True)

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

