# 🎧 Audio Signal Classification using ML & Deep Learning  
*Kaggle Competition Project*

## 📌 Project Overview
This project was developed as part of a Kaggle audio classification competition.  
The goal was to process raw audio signals, extract meaningful acoustic features, and build predictive models capable of accurately classifying audio samples into predefined categories.

The project combines both traditional Machine Learning techniques and Deep Learning approaches to evaluate performance differences and build a robust classification system.

---

## 🏆 Kaggle Competition Context
The competition required participants to:

- Process raw `.wav` audio files
- Perform time-domain and frequency-domain feature extraction
- Train classification models
- Optimize performance based on evaluation metrics
- Generate submission files for leaderboard ranking

---

## ⚙️ Technical Implementation

### 🔹 1️⃣ Machine Learning Pipeline

**Audio Processing**
- Signal loading using `librosa`
- Audio normalization & trimming
- Noise handling and signal cleaning
- Efficient data processing with `tqdm` for progress tracking

**Feature Engineering**
- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma Features
- Spectral Centroid
- Spectral Bandwidth
- Zero Crossing Rate
- Spectral Rolloff

**Models Used**
- Random Forest Classifier
- XGBoost Classifier

**Optimization Techniques**
- Feature scaling
- Cross-validation
- Hyperparameter tuning
- Model performance comparison

---

### 🔹 2️⃣ Deep Learning Pipeline

- Conversion of audio signals into Mel-Spectrograms
- CNN-based architecture using TensorFlow / Keras
- Regularization (Dropout, Batch Normalization)
- Validation-based model selection
- Performance comparison against ML models

---

## 📊 Results & Analysis
- Evaluated model performance using competition metrics
- Compared Random Forest, XGBoost, and CNN results
- Generated Kaggle-compatible submission files
- Observed differences between feature-engineered ML models and end-to-end deep learning models

---

## 🛠️ Tech Stack
- Python
- NumPy, Pandas
- Librosa (Audio Signal Processing)
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- tqdm (Pipeline Progress Monitoring)
- Matplotlib

---

## 🚀 Key Learnings
- Audio signal processing fundamentals
- Advanced feature engineering for acoustic data
- Ensemble learning with Random Forest & XGBoost
- CNN-based learning on spectrogram representations
- End-to-end Kaggle competition workflow
