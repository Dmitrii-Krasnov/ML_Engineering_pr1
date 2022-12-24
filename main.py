import streamlit as st
from transformers import pipeline

@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline(
        "summarization",
        "pszemraj/long-t5-tglobal-base-16384-book-summary"
    )

def load_text():
    generation_text = st.text_area(label='Enter text foe summarization')
    if generation_text:
        return generation_text
    else:
        return ''


st.title('Text summary generation')
st.markdown('<span style="color: red; font-weight: bold">NB!</span> English only supported', unsafe_allow_html=True)
txt = load_text()

result = st.button('Get summary')
if result:
    if len(txt) > 0:
        generator = load_model()
        with st.spinner('Generation in progress...'):
            summary = generator(txt)[0]['summary_text']
            st.write('**Results:**')
            st.write(summary)
    else:
        st.write('<span style="color: red">Provide some text for generation!</span>', unsafe_allow_html=True)
