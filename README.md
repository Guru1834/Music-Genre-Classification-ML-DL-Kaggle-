# 🎧 Audio Signal Classification using Machine Learning & Deep Learning  
*Kaggle Competition Project*

## 📌 Project Overview
This project was developed as part of a Kaggle audio classification competition.  
The objective was to analyze raw audio signals, extract meaningful acoustic features, and build predictive models to accurately classify audio samples into predefined categories.

The project includes both:
- A traditional Machine Learning pipeline
- A Deep Learning-based approach using neural networks

This allowed performance comparison between feature-based ML models and end-to-end DL models.

---

## 🏆 Kaggle Competition
This project was built for a Kaggle audio classification challenge where participants were required to:

- Process raw `.wav` audio files
- Perform feature extraction from time and frequency domains
- Train predictive models
- Optimize performance based on competition evaluation metrics
- Generate submission files for leaderboard ranking

---

## ⚙️ Technical Approach

### 🔹 1. Machine Learning Pipeline

**Audio Preprocessing**
- Noise handling
- Audio normalization
- Signal trimming

**Feature Engineering**
- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma features
- Spectral centroid
- Zero Crossing Rate
- Spectral bandwidth & rolloff

**Models Used**
- Random Forest Classifier
- XGBoost Classifier

**Techniques Applied**
- Feature scaling
- Cross-validation
- Hyperparameter tuning
- Model comparison

---

### 🔹 2. Deep Learning Pipeline

- Conversion of audio signals into Spectrograms / Mel-Spectrograms
- Convolutional Neural Network (CNN) implementation using TensorFlow / Keras
- Regularization techniques (Dropout, Batch Normalization)
- Model evaluation on validation set
- Performance comparison with ML models

---

## 📊 Results
- Compared performance of Random Forest, XGBoost, and CNN models
- Evaluated using competition metric
- Generated leaderboard submission files
- Analyzed strengths of feature-based vs deep learning approaches

---

## 🛠️ Tech Stack
- Python
- NumPy, Pandas
- Librosa (Audio Signal Processing)
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib

---

## 🚀 Key Learnings
- Audio signal processing fundamentals
- Feature engineering for sound classification
- Boosting algorithms (XGBoost) in structured audio features
- CNN-based learning on spectrogram images
- End-to-end ML pipeline development for Kaggle competitions

