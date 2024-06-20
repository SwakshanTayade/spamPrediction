import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# to run streamlit run app.py


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    # remove the special chars
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    # cloning
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    # cloning
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Message Spam Detector")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1 preprocess
    transformed_sms = transform_text(input_sms)
    # 2 vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3 prediction
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")
