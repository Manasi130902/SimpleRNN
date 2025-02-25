import  numpy as np
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


#load the imdb datasel word_index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}




#load the pretrained model with relu activation function
model=load_model('simple_rnn_imdb.h5')



#helper functions -1
#function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])




#function to preprocess user input

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review= sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review




#design my streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

#user input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)



    #make prediction
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'


    #display the result
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction Score:{prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
    
    