import streamlit as st
import torch
from transformers import pipeline

@st.cache(allow_output_mutation=True)
def load_model(len):
    return pipeline(
        "summarization",
        "pszemraj/long-t5-tglobal-base-16384-book-summary",
        max_length=len,
        device=0 if torch.cuda.is_available() else -1,
    )

def load_text():
    generation_text = st.text_area(label='Enter text for summarization')
    if generation_text:
        return generation_text
    else:
        return ''


def load_len():
    generation_len_txt = st.text_input(label='Enter generated text max length')
    if generation_len_txt:
        try:
            generation_len = int(generation_len_txt)
        except:
            generation_len = 0
        return generation_len
    else:
        return 0


st.title('Text summary generation')
st.markdown('<span style="color: red; font-weight: bold">NB!</span> English only supported', unsafe_allow_html=True)
txt = load_text()
txt_len = load_len()

result = st.button('Get summary')
if result:
    if len(txt) > 0 and txt_len > 0:
        with st.spinner('Generation in progress...'):
            generator = load_model(txt_len)
            summary = generator(txt)[0]['summary_text']
            st.write('**Results:**')
            st.write(summary)
    else:
        st.write('<span style="color: red">Provide some text and text length for generation!</span>', unsafe_allow_html=True)
