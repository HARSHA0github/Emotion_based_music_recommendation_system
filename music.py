import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
from PIL import Image
import io
import pywhatkit
import os
import random

from mediapipe.tasks.python.vision import HolisticLandmarker
from mediapipe.tasks.python.vision import HolisticLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python import BaseOptions

# ── Load model & labels 
@st.cache_resource
def load_assets():
    mdl  = load_model("model.h5")
    lbl  = np.load("labels.npy")
    opts = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="holistic_landmarker.task"),
        running_mode=RunningMode.IMAGE,
    )
    hl = HolisticLandmarker.create_from_options(opts)
    return mdl, lbl, hl

model, label, holis = load_assets()

# ── Page 
st.header("Emotion Based Music Recommender")

# ── Session state init 
if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""
if "captured" not in st.session_state:
    st.session_state["captured"] = False

# ── Emotion-to-Audio Mapping 
EMOTION_MAPPING = {
    "Happy": "upbeat pop feel good",
    "Sad": "acoustic melancholic slow",
    "Angry": "rock heavy metal intense",
    "Surprise": "electronic fast paced synth",
    "Neutral": "lo-fi chill ambient",
    "Rock": "rock intense"
}

# ── Singer Suggestions Dictionary 
SINGER_SUGGESTIONS = {
    "english": "Taylor Swift, Ed Sheeran, Ariana Grande, Drake, The Weeknd, Justin Bieber, Billie Eilish, Dua Lipa, Eminem, Bruno Mars",
    "hindi": "Arijit Singh, Shreya Ghoshal, Kishore Kumar, Lata Mangeshkar, Udit Narayan, Neha Kakkar, Jubin Nautiyal, Sonu Nigam, Atif Aslam, Kumar Sanu",
    "spanish": "Bad Bunny, J Balvin, Shakira, Rosalía, Maluma, Daddy Yankee, Enrique Iglesias, Luis Fonsi, Ozuna, Karol G",
    "korean": "BTS, BLACKPINK, IU, EXO, TWICE, Stray Kids, SEVENTEEN, Red Velvet, NCT, TXT",
    "tamil": "A.R. Rahman, Anirudh Ravichander, S.P. Balasubrahmanyam, Sid Sriram, K.S. Chithra, Yuvan Shankar Raja, Hariharan, Shreya Ghoshal, Vijay Prakash, Karthik",
    "telugu": "S.P. Balasubrahmanyam, Sid Sriram, K.S. Chithra, Mangli, Ram Miriyala, Anurag Kulkarni, Geetha Madhuri, Sunitha, S. Janaki, Karthik"
}

# ── Inputs 
lang = st.text_input("Language :")
st.caption("Suggestions: English, Hindi, Spanish, Korean, Tamil, Telugu")

singer = ""
if lang:
    singer = st.text_input("Singer :")
    lang_key = lang.strip().lower()
    if lang_key in SINGER_SUGGESTIONS:
        st.caption(f"Top 10 Singers: {SINGER_SUGGESTIONS[lang_key]}")
    else:
        st.caption("Enter the name of your favorite singer.")

img_file = None
if lang and singer:
    # ── Camera capture 
    st.markdown("#### Capture your emotion")
    
    if st.session_state["captured"] and st.session_state["emotion"]:
        st.success(f"Emotion already captured: **{st.session_state['emotion']}** (You can retake if you want)")
        
    st.info("Click **'Take Photo'** — face the camera clearly, then capture.")

    img_file = st.camera_input("Take Photo", key="camera")

if img_file is not None:
    # Convert to OpenCV BGR
    pil_img = Image.open(io.BytesIO(img_file.getvalue())).convert("RGB")
    frm     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frm     = cv2.flip(frm, 1)
    frm = cv2.resize(frm, (640, 480))

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    res = holis.detect(mp_image)

    lst = []

    if res.face_landmarks:
        face_lms = res.face_landmarks          # list of NormalizedLandmark

        for lm in face_lms:
            lst.append(lm.x - face_lms[1].x)
            lst.append(lm.y - face_lms[1].y)

        if res.left_hand_landmarks:
            left_lms = res.left_hand_landmarks
            for lm in left_lms:
                lst.append(lm.x - left_lms[8].x)
                lst.append(lm.y - left_lms[8].y)
        else:
            lst.extend([0.0] * 42)

        if res.right_hand_landmarks:
            right_lms = res.right_hand_landmarks
            for lm in right_lms:
                lst.append(lm.x - right_lms[8].x)
                lst.append(lm.y - right_lms[8].y)
        else:
            lst.extend([0.0] * 42)

        arr  = np.array(lst).reshape(1, -1)
        arr = arr[:, :1020]
        pred = label[np.argmax(model.predict(arr))]
        # The model's labels array has some typos ('susurprise' and 'rocrock')
        if pred == "susurprise":
            pred = "surprise"
        elif pred == "rocrock":
            pred = "rock"

        # Annotate and display
        cv2.putText(frm, str(pred), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        st.image(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB),
                 caption=f"Detected emotion: {pred}",
                 use_container_width=True)

        st.session_state["emotion"]  = str(pred)
        st.session_state["captured"] = True

    else:
        st.warning("No face detected — ensure good lighting and face the camera directly.")
        st.session_state["captured"] = False

# ── Emotion status banner 
if not (st.session_state["captured"] and st.session_state["emotion"]) and (lang and singer):
    pass # Already shown success message near camera if captured

# ── Recommend button 
if lang and singer:
    st.divider()
    if st.button("Recommend me songs", type="primary"):
        if not lang or not singer:
            st.warning("Please fill in both Language and Singer fields.")
        elif not st.session_state["emotion"]:
            st.warning("Please capture your emotion first using the camera above.")
        else:
            emotion = st.session_state["emotion"]
            emotion_key = str(emotion).strip().capitalize()
            mapped_query = EMOTION_MAPPING.get(emotion_key, emotion)
            
            # Construct the final search text based on the mapped query, lang, and singer
            search_query = f"{lang} {mapped_query} song {singer}"
            
            st.markdown(f"Opening YouTube for: **{search_query}**")
            
            # Youtube Autoplay using approach B directly
            try:
                pywhatkit.playonyt(search_query)
            except Exception as e:
                st.error(f"Could not open YouTube automatically: {e}")
                url = (f"https://www.youtube.com/results?search_query="
                       f"{lang}+{mapped_query.replace(' ', '+')}+song+{singer}")
                webbrowser.open(url)
                st.markdown(f"[Click here if YouTube did not open automatically]({url})")
            
            # Local Playback
            st.divider()
            st.markdown(f"### Play from Local Storage ({emotion_key})")
            
            local_folder = os.path.join("local_music", emotion_key.lower())
            if os.path.exists(local_folder):
                songs = [f for f in os.listdir(local_folder) if f.endswith(('.mp3', '.wav', '.ogg', '.m4a', '.webm'))]
                if songs:
                    selected_song = random.choice(songs)
                    song_path = os.path.join(local_folder, selected_song)
                    st.success(f"Playing a local **{emotion_key}** song: `{selected_song}`")
                    st.audio(song_path)
                else:
                    st.warning(f"No audio files (.mp3, .wav) found in the `{local_folder}` folder.")
            else:
                st.warning(f"Local folder not found: `{local_folder}`")
