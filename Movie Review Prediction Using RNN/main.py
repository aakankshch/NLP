#Import Libraries
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.utils import pad_sequences 
from keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Function for preprocessing
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')


