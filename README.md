# EMG Gesture Classification HMI System
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/machine%20learning-SVM%20%7C%20KNN%20%7C%20RF-orange)
![Signal Processing](https://img.shields.io/badge/signal%20processing-EMG-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-used-yellow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Overview
This project implements an EMG (Electromyography) gesture classification pipeline along with an interactive Human-Machine Interface built using Streamlit.

The system:
- Loads EMG data from `.mat` files
- Segments signals using sliding windows
- Extracts time-domain features
- Trains multiple machine learning models
- Evaluates performance using standard metrics
- Provides an interactive interface for real-time testing

---

## Project Structure

```
project/
│
├── src/
│   ├── main.py        # Training pipeline
│   └── app.py         # Streamlit HMI application
│
├── data/
│   └── DB2/...        # Dataset (not included)
│
├── results/
│   ├── svm.pkl
│   ├── knn.pkl
│   ├── rf.pkl
│   └── scaler.pkl
│
└── README.md
```

---

## Features

### Signal Processing
- Sliding window segmentation
- Majority voting for window labels

### Feature Extraction
For each EMG channel:
- Mean Absolute Value (MAV)
- Root Mean Square (RMS)
- Waveform Length (WL)

### Machine Learning Models
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest

### Evaluation Metrics
- Accuracy
- Confusion Matrix
- Classification Report
- Per-class F1-score

### Visualization
- Model accuracy comparison
- Confusion matrix heatmap
- Per-class F1 score bar chart

### HMI (Streamlit App)
- Upload `.mat` files
- Select model dynamically
- Adjust window & step size
- View predictions and performance

---

## Installation

### 1. Clone the repository
```
git clone https://github.com/stasinosntaveas/emg-signal-classification-hmi.git
cd emg-signal-classification-hmi
```

### 2. Create virtual environment (optional but recommended)
```
python -m venv venv
```

#### Activate it:

Linux/macOS:
```
source venv/bin/activate
```

Windows:
```
venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

---

## Dataset

This project uses the **Ninapro DB2 dataset**, which contains:

- 40 intact subjects
- 12-channel surface EMG recordings
- 49 hand movements + rest condition
- Sampling rate: 2 kHz
- Synchronized labels (stimulus / restimulus)

Dataset link:
[Ninapro DB2 Dataset](https://ninapro.hevs.ch/instructions/DB2.html)

Reference:
Atzori et al., *Electromyography data for non-invasive naturally-controlled robotic hand prostheses*, Scientific Data (2014)
[Read the paper](https://www.nature.com/articles/sdata201453)

---

## Usage

### 1. Train Models
Run:
```
cd src
python main.py
```

This will:
- Train all models
- Print evaluation results
- Generate plots
- Save trained models in `/results`

---

### 2. Run HMI Application

```
cd src
streamlit run app.py
```

Then open the provided local URL in your browser.

---

## HMI Controls

- **Model Selection**: Choose between SVM, KNN, Random Forest
- **Window Size**: Adjust segmentation size
- **Step Size**: Control overlap between windows
- **Manual Overrides**: Fine-tune parameters

---

## Output

The application provides:
- Real-time accuracy
- Confusion matrix visualization
- Sample predictions

---

## Model Saving

After training, the following files are stored:

- `svm.pkl`
- `knn.pkl`
- `rf.pkl`
- `scaler.pkl`

These are automatically loaded by the Streamlit app.

<!-- ---

## Future Improvements

- Add frequency-domain features
- Deep learning models (CNN/LSTM)
- Real-time EMG acquisition support
- Hyperparameter tuning
- Cross-validation -->

---

## Notes

- Ensure dataset paths are correct before running
- Large window sizes may slow down processing
- Model performance depends heavily on preprocessing and feature quality

---

## License

This project is licensed under the MIT License.
