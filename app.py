from predictor import predict, load_model
import streamlit as st
import json, uuid, asyncio

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

firebase_login_details = {
  "type": st.secrets['TYPE'],
  "project_id": st.secrets['PROJECT_ID'],
  "private_key_id": st.secrets['PRIVATE_KEY_ID'],
  "private_key": st.secrets['PRIVATE_KEY'],
  "client_email": st.secrets['CLIENT_EMAIL'],
  "client_id": st.secrets['CLIENT_ID'],
  "auth_uri": st.secrets['AUTH_URI'],
  "token_uri": st.secrets['TOKEN_URI'],
  "auth_provider_x509_cert_url": st.secrets['AUTH_PROVIDER_CERT_URL'],
  "client_x509_cert_url": st.secrets['CLIENT_CERT_URL'],
}

# save to file
with open('firebase_login.json', 'w+') as output:
	# dump to JSON
	json.dump(firebase_login_details, output)

# login to firebase
# must check if already initialized
# somehow streamlit loves to rerun the
# code. idk why
if not firebase_admin._apps:
	cred = credentials.Certificate('firebase_login.json')
	firebase_admin.initialize_app(cred)

db = firestore.client()

# set page config
st.set_page_config(
	page_title="Analyze Your Hotel Review",
	page_icon="🏨"
)

# load model
with st.spinner("Loading our awesome AI 🤩. Please wait ..."):
	model = load_model()

@st.cache
def handle_text(text):
	# predict
	prediction = predict(model, text)

	# save to firebase
	saveToFirebase(text, prediction)

	# return
	return prediction

def saveToFirebase(text, prediction):
	# convert to 0 (negative) or 1 (positive)
	prediction = 1 if prediction > 50 else 0

	db.collection('prediction').add({
		'id': str(uuid.uuid4()),
		'text': text,
		'prediction': prediction
	})

# title and subtitle
st.title("🏨 Hotel Review Sentiment Analysis")
st.write("Do you think that your customer loves your hotel? Do they love the facility you gave? 🛎️")
st.write("Checking all the review is not an easy task, let our AI do it for you! 😆")
st.write("It's easy and fast. Put the review down below and we will take care the rest 😉")

# user input
user_review = st.text_area(
	label="Review:",
	help="Input your (or your client's) review here, then click anywhere outside the box."
)

if user_review != "":
	prediction = handle_text(user_review)

	# display prediction
	st.subheader("AI thinks that ...")

	# check prediction
	if prediction > 50:
		st.write(f"YAY! It's a positive review 🥰🥰. We're {round(prediction, 3)}% sure it's a positive review")
	else:
		st.write(f"NOOOO! It's a negative review 😱😱. We're {round(100-prediction, 3)}% sure it's a negative review")