import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import PyPDF2
import torch
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Streamlit UI with dark theme style
st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="centered")

# Load model and tokenizer once at startup
@st.cache_resource
def load_model_tokenizer(model_dir="./t5_finetuned_full_data"):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return tokenizer, model

tokenizer, model = load_model_tokenizer()

# Extract text from uploaded PDF file
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + " "
    return text.strip()

# Summarize text
def summarize_text(text, max_input_length=512, max_summary_ratio=0.75,min_summary_ratio=0.2):
    # Tokenize input once to find token count
    inputs = tokenizer.encode(text, max_length=max_input_length, truncation=True)
    input_token_count = len(inputs)
    # Set max_summary_length to 50% and min to 30%of input tokens
    max_length = max(1, int(input_token_count * max_summary_ratio))
    min_length = max(1, int(input_token_count * min_summary_ratio))
    # Generate summary ids
    inputs = torch.tensor([inputs])
    with torch.no_grad():
        summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length ,num_beams=4, early_stopping=True)
        
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary




dark_theme_css = """
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #1f2937;
        color: white;
    }
    .stFileUploader>div>div>input {
        color: white;
        background-color: #1f2937;
    }
</style>
"""
st.markdown(dark_theme_css, unsafe_allow_html=True)

st.title("PDF Document Summarizer")
st.write("Upload a PDF file, and get a concise summary!")

uploaded_file = st.file_uploader("Choose a PDF file to summarize", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
    
    st.subheader("Extracted Text")
    st.text_area("", raw_text, height=200)
    
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = summarize_text(raw_text)
        st.subheader("Summary")
        st.write(summary)
