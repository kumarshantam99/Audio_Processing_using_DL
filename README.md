# Audio_Processing_using_DL

Audio Analysis System
A deep learning-based system for analyzing audio characteristics including tone, pitch, and pace. This system provides detailed analysis of audio files and can be trained on various datasets.

## Features

* Audio feature extraction (pitch, spectral characteristics, MFCCs)
* Deep learning model for audio analysis
* Support for multiple audio datasets (RAVDESS, TESS)
* Comprehensive score interpretation
* Model training and evaluation capabilities

## Installation
### Prerequisites

```bash
pip install librosa torch numpy pandas soundfile tqdm
```

## Supported Datasets

### RAVDESS Dataset

* Contains emotional speech recordings
* Available at: https://zenodo.org/record/1188976
* Features controlled variations in pitch, tone, and emotional intensity

### TESS Dataset

* Toronto Emotional Speech Set
* Available at: https://tspace.library.utoronto.ca/handle/1807/24487
* High-quality emotional speech samples

## Core Components
1. Feature Extraction (AudioFeatureExtractor)
Extracts relevant features from audio files:

* Pitch characteristics (mean and standard deviation)
* Spectral features (centroids and rolloff)
* MFCCs (Mel-frequency cepstral coefficients)
* Tempo analysis

2. Neural Network Model (AudioAnalysisModel)
A deep neural network for audio analysis:

* Three-layer architecture
* ReLU activation functions
* Dropout regularization
* Outputs three scores: tone, pitch, and pace

```architecture
Input Layer (18 features)
    ↓
Dense Layer (128 units) + ReLU + Dropout(0.2)
    ↓
Dense Layer (64 units) + ReLU + Dropout(0.2)
    ↓
Dense Layer (32 units) + ReLU
    ↓
Output Layer (3 scores)
```

3. Label Extraction (LabelExtractor)
Handles label extraction from different dataset formats:

* RAVDESS format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
* TESS format: [OY]AF_emotion_word.wav
* Custom format: tone_X_pitch_Y_pace_Z.wav

4. Result Interpretation
Provides human-readable analysis of results:

* Scores scaled from 1-10
* Descriptive interpretations
* Technical measurements
* Audio classification suggestions

## Interpretation Guidelines
### Tone Score

* 8.5-10: Exceptionally clear and pure
* 7-8.4: Very clear and clean
* 5.5-6.9: Moderately clear
* 4-5.4: Slightly unclear
* 1-3.9: Unclear or noisy

### Pitch Score

* 8.5-10: Very high
* 7-8.4: High
* 5.5-6.9: Medium-high
* 4-5.4: Medium-low
* 1-3.9: Low

### Pace Score

* 8.5-10: Very fast
* 7-8.4: Fast
* 5.5-6.9: Moderate to fast
* 4-5.4: Moderate
* 1-3.9: Slow

## Limitations

* Requires high-quality audio input
* Best suited for speech and musical content
* Model performance depends on training data quality
* May need retraining for specific use cases
