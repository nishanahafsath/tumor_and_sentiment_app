import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK data
nltk.download('vader_lexicon')

# Load the saved models
loaded_tumor_model = tf.keras.models.load_model("tumor_detection.h6")
loaded_dnn_model = tf.keras.models.load_model("dnn_model.h6")
loaded_rnn_model = tf.keras.models.load_model("rnn_model.h5")
loaded_lstm_model = tf.keras.models.load_model("lstm_model.h5")
loaded_backpropagation_model = tf.keras.models.load_model("backpropagation_model.h5")

def make_prediction_cnn(uploaded_file, model):
    content = uploaded_file.read()
    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    res = model.predict(img)
    if res > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor"

def make_sentiment_prediction(user_input):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(user_input)
    
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def main():
    st.title("Sentimental Analysis and Brain Tumor Detection App")

    task_options = ["Tumor Detection", "Sentiment Analysis"]
    selected_task = st.selectbox("Choose a task:", task_options)

    if selected_task == "Sentiment Analysis":
        sentiment_analysis()
    elif selected_task == "Tumor Detection":
        tumor_detection()

def sentiment_analysis():
    st.header("Sentiment Analysis Model Selection")
    model_options = ["Perceptron", "Backpropagation", "DNN", "RNN", "LSTM"]
    selected_model = st.selectbox("Choose a model:", model_options)
    
    st.write("Provide input data for Sentiment Analysis:")
    user_input = st.text_input("Enter text for analysis:")

    if st.button("Predict"):
        prediction = ""

        if selected_model == "Perceptron":
            prediction = make_sentiment_prediction(user_input)
        elif selected_model == "Backpropagation":
            prediction = make_sentiment_prediction(user_input)
        elif selected_model == "DNN":
            prediction = make_sentiment_prediction(user_input)
        elif selected_model == "RNN":
            prediction = make_sentiment_prediction(user_input)
        elif selected_model == "LSTM":
            prediction = make_sentiment_prediction(user_input)

        st.write(f"Prediction: {prediction}")

def tumor_detection():
    st.header("Tumor Detection Model Selection")
    model_options = ["CNN"]
    selected_model = st.selectbox("Choose a model:", model_options)
    
    st.write("Provide input data for Tumor Detection:")
    uploaded_file = st.file_uploader("Choose an image for prediction", type=["jpg", "png"])

    if uploaded_file is not None:
        if selected_model == "CNN":
            prediction = make_prediction_cnn(uploaded_file, loaded_tumor_model)

            # Display the result
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()

