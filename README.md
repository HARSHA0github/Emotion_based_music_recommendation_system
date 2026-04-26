# Emotion-Based Music Recommender

An intelligent system that detects user emotions using facial landmarks and recommends music accordingly in real time. The project combines computer vision, machine learning, and a web-based interface to deliver a personalized music experience.

---

## Overview

This project captures a user's facial expression through a webcam, extracts facial and hand landmarks using MediaPipe, and predicts the user's emotion using a trained machine learning model. Based on the detected emotion, the system recommends music via YouTube or local storage.

---

## Features

- Real-time emotion detection using webcam  
- Landmark-based feature extraction (MediaPipe Holistic)  
- Emotion classification using trained model (`.h5`)  
- Music recommendation based on detected emotion  
- Supports both **online (YouTube)** and **offline (local songs)** playback  
- User input for language and singer preference  
- Interactive UI using Streamlit  

---

## Tech Stack

- **Frontend**: Streamlit  
- **Computer Vision**: OpenCV, MediaPipe  
- **Machine Learning**: TensorFlow / Keras  
- **Programming Language**: Python  
- **Music Playback**: PyWhatKit, Webbrowser  

---

## Project Structure
Emotion-Music-Recommender/
│── app.py
│── model.h5
│── labels.npy
│── holistic_landmarker.task
│── local_music/
│ ├── happy/
│ ├── sad/
│ ├── angry/
│── requirements.txt
│── README.md


---

##  Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/emotion-music-recommender.git
cd emotion-music-recommender

2.Install dependencies:
pip install -r requirements.txt

3.Run the application:
streamlit run app.py

---

## How It Works
Capture image from webcam

Extract facial & hand landmarks using MediaPipe

Convert landmarks into feature vector

Predict emotion using trained ML model

Map emotion to music category

Play music (YouTube or local storage)

## Supported Emotions
Happy

Sad

Angry

Surprise

Neutral

Rock

#Model Details
Type: Supervised Learning

Architecture: Neural Network (Keras)

Input: Landmark-based feature vector

Output: Emotion class (Softmax)

#Limitations
Single-frame emotion detection (no temporal tracking)

Rule-based music recommendation

Performance depends on lighting and camera quality

## Future Improvements:
Real-time continuous emotion tracking

Integration with Spotify / JioSaavn APIs

Personalized recommendation using user history

Multimodal emotion detection (voice + text)

## Contributing
Contributions are welcome. Feel free to fork the repository and submit a pull request.

## License
This project is open-source and available under the MIT License.

## Author
Y Harsha Vardhan
CHL Pramad
B.E CSE (AIML)
