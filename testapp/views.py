from django.shortcuts import render
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model
import numpy as np


model = load_model('testapp/ML_model/Text_model')

with open('testapp/ML_model/tokenizer.pickle', 'rb') as handle:
    token = pickle.load(handle)

def home(request):
    if request.method == 'POST':
        n_words = int(request.POST.get('num_words'))
        initial_lyrics = request.POST.get('initial_lyrics')
        lyrics = generate_songs(model , n_words , initial_lyrics)
        return render(request , 'testapp/home.html' ,{'lyrics': lyrics})
    return render(request , 'testapp/home.html')

def generate_songs(model, n_words , seed_text):
    for i in range(n_words):
        tokenized_sentence = token.texts_to_sequences([seed_text])
        padded_sequence = pad_sequences(tokenized_sentence , maxlen= model.input_shape[1])
        prediction_index = np.argmax(model.predict(padded_sequence , verbose = 0))
        predicted_word = token.index_word[prediction_index]
        seed_text += ' ' + predicted_word
        
    return seed_text