import tensorflow as tf
import numpy as np
import streamlit as st
import os

# ML stuff
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras

# preprocessing library
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import normalize_whitespace, lower_text, remove_eol_characters, replace_currency_symbols, \
                                        remove_punct, remove_multiple_spaces_and_strip_text, filter_non_latin_characters

GOOGLE_DRIVE_FILE_ID = "1mHf5gAywjFapLXZa_noeC3vW0pviAYhB"

# set maximum length and tokenizer
MAX_LEN = 50
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-lite-base-p1')

# stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# stopword
stopword_factory = StopWordRemoverFactory()
stopword = stopword_factory.create_stop_word_remover()

# use nlpretext processor
preprocessor = Preprocessor()
preprocessor.pipe(lower_text)
preprocessor.pipe(remove_eol_characters)
preprocessor.pipe(normalize_whitespace)
preprocessor.pipe(remove_multiple_spaces_and_strip_text)
preprocessor.pipe(remove_punct)
preprocessor.pipe(replace_currency_symbols)
preprocessor.pipe(filter_non_latin_characters)

# load model on first launch
@st.cache(allow_output_mutation=True)
def load_model():
	# path to file
	filepath = "model/model.h5"

	# folder exists?
	if not os.path.exists('model'):
		# create folder
		os.mkdir('model')
	
	# file exists?
	if not os.path.exists(filepath):
		# download file
		from gd_download import download_file_from_google_drive
		download_file_from_google_drive(id=GOOGLE_DRIVE_FILE_ID, destination=filepath)
	
	# load model
	model = keras.models.load_model(filepath)
	return model

def cleanText(sentence):
    # process with PySastrawi first
    stemmed = stemmer.stem(sentence)
    stopwordremoved = stopword.remove(stemmed)

    # then with nlpretext
    cleaned = preprocessor.run(stopwordremoved)

    # return
    return cleaned

def encodeText(sentence):
	sentence = cleanText(sentence)

	encoded_dict = tokenizer.encode_plus(
					sentence,
					add_special_tokens = True,
                    max_length = MAX_LEN,
                    truncation = True,
                    padding = "max_length",
                    return_attention_mask = True,
                    return_token_type_ids = False
	)

	input_ids = [encoded_dict['input_ids']]
	attn_mask = [encoded_dict['attention_mask']]
  	
	return input_ids, attn_mask

def predict(model, input):
	input_id, attn_mask = np.array(encodeText(input))
	data = [input_id, attn_mask]

	prediction = model.predict(data)
	prediction = prediction[0].item() * 100

	return prediction