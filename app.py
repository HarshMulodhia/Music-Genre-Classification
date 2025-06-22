import streamlit as st
from keras.models import load_model
import numpy as np
import librosa
import pandas as pd
import joblib
from streamlit_advanced_audio import audix, WaveSurferOptions
import matplotlib.pyplot as plt
from PIL import Image

st.title("Music Genre Classifier")
st.write("This app uses Convolutional Neural Network Model to classify music genre of a song based on its audio features.")

st.header('Select Song:')
uploaded_file = st.file_uploader("Upload an audio file to predict their genre:", type=["wav"])

if uploaded_file is not None:
    audio_array, sampling_rate = librosa.load(uploaded_file)
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sampling_rate, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Creating a dictionary to hold features
    features = {}
    for i in range(len(mfcc_mean)):
        features[f'mfcc_mean_{i+1}'] = mfcc_mean[i]
        features[f'mfcc_std_{i+1}'] = mfcc_std[i]

    # Converting to a pandas Series
    x_series = pd.Series(features)


    # Preprocessing 
    scalar = joblib.load('standard_scalar.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    x_scaled = scalar.fit_transform(pd.DataFrame(x_series))

    # Loading the Trained Model
    model = load_model('music_genre_classification_model.keras')
    # Making Prediction
    y_pred = model.predict(x_scaled[np.newaxis, :, :])
    y_pred_class = np.argmax(y_pred, axis=1)

    predicted_genre = label_encoder.inverse_transform(y_pred_class)

    st.subheader('Waveform Audio Player:')
    options = WaveSurferOptions(
        wave_color="#94FFD2",
        height=80,
    )
    result = audix(uploaded_file, wavesurfer_options=options)

    # Computing the STFT coefficients for spectrogram visualization
    st.header('Spectrogram Visualization for the given uploaded audio file:')
    d = librosa.stft(audio_array)
    # Converting to dB
    d_db = librosa.amplitude_to_db(np.abs(d), ref=np.max)

    fig_spec = plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(d_db, sr=sampling_rate, x_axis='time', y_axis='hz')
    fig_spec.colorbar(img, format='%+2.0f dB')
    st.pyplot(fig_spec)

    st.header('Genre Prediction based on the trained CNN Model:')
    st.write('The model predicted the genre of the song to be:',predicted_genre[0].capitalize())


    st.header('Model Evaluation:')
    st.markdown('''
        - The Model attained an **Overall Accuracy** of 98.02% and a **Validation Accuracy** of 61.5%.
        - The plot below depicts the variation in Accuracy of the Model:''')
    image = Image.open('Training and Validation Accuracy.png')
    st.image(image, caption='Training and Validation Accuracy')

    st.markdown('''
        - The Model has a **Training Loss** of 6.59% and a **Validation Loss** of 6.25%.
        - The plot below depicts the variation in Losses of the Model:''')
    image = Image.open('Training and Validation Loss.png')
    st.image(image, caption='Training and Validation Loss')

    st.markdown('''
        - The plot below depicts the Confusion Matrix of the Genre predictions by the Model:''')
    image = Image.open('Confusion Matrix.png')
    st.image(image, caption='Confusion Matrix')

    st.markdown('''
        - The plot below depicts the Genres with their confidence score:''')
    image = Image.open('Predicted Genres with Confidence Scores.png')
    st.image(image, caption='Predicted Genres with Confidence Scores')



