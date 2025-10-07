# ECG Arrhythmia Classification using DL Models

This project implements a **deep learning model using LSTM networks** to classify ECG heartbeats from the **MIT-BIH Arrhythmia Database** into different arrhythmia categories.

---

## Overview

- **Goal:** Automatic classification of ECG heartbeats into arrhythmia types.  
- **Model:** Stacked LSTM architecture for multi-class classification.  
- **Processing:** Automated ECG preprocessing — filtering, R-peak detection, segmentation, normalization.  
- **Evaluation:** Includes metrics and visualizations.  
- **Accuracy:** Achieves around **98% test accuracy** on MIT-BIH dataset.

---

## Dataset

**MIT-BIH Arrhythmia Database (PhysioNet)**  
- 48 half-hour two-channel ECG recordings  
- Sampling Rate: 360 Hz  
- Expert-annotated beats  

**Classes:**
- N — Normal beats  
- S — Supraventricular ectopic beats  
- V — Ventricular ectopic beats  
- F — Fusion beats  
- Q — Unknown/Other beats  

---

## Model Architecture
```
Input (250, 1)
↓
LSTM(128, return_sequences=True)
↓
Dropout + BatchNormalization
↓
LSTM(64, return_sequences=True)
↓
Dropout + BatchNormalization
↓
LSTM(32)
↓
Dropout + BatchNormalization
↓
Dense(64, ReLU) → Dense(32, ReLU)
↓
Dense(num_classes, Softmax)
```

**Training Details:**
- Optimizer: Adam (lr = 0.001)  
- Loss: Categorical Crossentropy  
- Batch Size: 128  
- Max Epochs: 100  
- Early Stopping and Learning Rate Scheduling enabled  

---

## Performance

| Metric | Value |
|--------|--------|
| Accuracy | ~98% |
| Precision | ~97% |
| Recall | ~96% |
| F1-Score | ~97% |

**Generated Visualizations:**
- Training history (accuracy/loss)
- Confusion matrices
- Sample predictions with confidence scores
- Raw vs filtered signal comparisons

---

## Workflow

1. **Preprocessing** — Run:
   ```bash
   python ecg_preprocessing.ipynb

   ```
   - Downloads MIT-BIH data
   - Filters noise
   - Segments beats and normalizes them
  
2. **Training** — Run:

   ```python
    lstm_model.ipynb

   ```
  - Trains the LSTM model
  - Saves best and final model checkpoints
    
3. **Prediction Example:**
   ```python
   from tensorflow import keras
   import numpy as np
    
   model = keras.models.load_model('best_lstm_model.keras')
   data = np.load('ecg_mitdb_processed.npz')
   preds = model.predict(data['X'][:10])
   classes = np.argmax(preds, axis=1)
   ```

   ## Project Structure
    ``` ecg-arrhythmia-classification/
    ├── ecg_preprocessing.ipynb
    ├── lstm_model.py
    ├── requirements.txt
    ├── README.md
    ├── data
       ├── ecg_mitdb_processed.npz
       ├── mitdb
    ├── final_lstm_model.keras
    ├── best_lstm_model.keras
    ├── evaluation_results.json
    └── *.png (visualizations)
    ```
    ## Requirements
   ```
   numpy>=1.21.0
   matplotlib>=3.4.0
   scipy>=1.7.0
   scikit-learn>=0.24.0
   tensorflow>=2.8.0
   wfdb>=4.0.0
   seaborn>=0.11.0
   tqdm>=4.62.0
   ```
   


   




