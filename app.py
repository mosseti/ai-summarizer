import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Text Summarizer", layout="centered")

st.title("AI Text Summarizer")
st.write("Paste any text and get an instant AI-generated summary.")

@st.cache_resource
def load_summarizer():
    # Smaller, faster model
    return pipeline("summarization", model="t5-small", tokenizer="t5-small")

summarizer = load_summarizer()

text = st.text_area("Enter text to summarize", height=250)

if st.button("Summarize"):
    if len(text.strip()) == 0:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Summarizing..."):
            result = summarizer(
                text,
                max_length=120,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]
        st.success("Summary:")
        st.write(result)
