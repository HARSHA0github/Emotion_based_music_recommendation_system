#  Emotion-Based Music Recommender

An intelligent real-time system that detects user emotions using facial landmarks and recommends music accordingly. The project combines computer vision, machine learning, and an interactive web interface to deliver a personalized music experience.

---

##  Overview

This project captures a user's facial expression through a webcam, extracts facial and hand landmarks using MediaPipe, and predicts the user's emotion using a trained machine learning model. Based on the detected emotion, the system dynamically recommends music through YouTube or local storage.

---

##  Features

- Real-time emotion detection using webcam  
- Landmark-based feature extraction (MediaPipe Holistic)  
- Emotion classification using trained model (`.h5`)  
- Emotion-aware music recommendation  
- Supports both online (YouTube) and offline (local songs) playback  
- User input for language and singer preference  
- Interactive UI using Streamlit  

---

##  Tech Stack

- Frontend: Streamlit  
- Computer Vision: OpenCV, MediaPipe  
- Machine Learning: TensorFlow / Keras  
- Programming Language: Python  
- Music Playback: PyWhatKit, Webbrowser  

---

##  Project Structure

Emotion-Music-Recommender/
│── app.py
│── model.h5
│── labels.npy
│── holistic_landmarker.task
│── local_music/
│   ├── happy/
│   ├── sad/
│   ├── angry/
│── requirements.txt
│── README.md

---

##  Installation

1. Clone the repository:
git clone https://github.com/your-username/emotion-music-recommender.git  
cd emotion-music-recommender  

2. Install dependencies:
pip install -r requirements.txt  

3. Run the application:
streamlit run app.py  

---

##  How It Works

1. Capture image from webcam  
2. Extract facial & hand landmarks using MediaPipe  
3. Convert landmarks into a feature vector  
4. Predict emotion using trained ML model  
5. Map emotion to music category  
6. Play music (YouTube or local storage)  

---

##  Supported Emotions

Happy, Sad, Angry, Surprise, Neutral, Rock  

---

##  Model Details

Type: Supervised Learning  
Architecture: Neural Network (Keras)  
Input: Landmark-based feature vector  
Output: Emotion class (Softmax)  

---

##  Limitations

- Single-frame emotion detection  
- Rule-based recommendation  
- Sensitive to lighting conditions  

---

##  Future Improvements

- Continuous emotion tracking  
- Spotify / JioSaavn integration  
- Personalized recommendation system  
- Multimodal emotion detection  

---

##  Contributing

Contributions are welcome. Fork and submit a pull request.

---

##  License

MIT License  

---

##  Authors

Y Harsha Vardhan  
CHL Pramad  
B.E CSE (AIML)
