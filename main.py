import streamlit as st

import utils


def load_text():
    generation_text = st.text_area(label='Enter text for summarization')
    if generation_text:
        return generation_text
    else:
        return ""


def load_len():
    generation_len_txt = st.text_input(label='Enter max length')
    if generation_len_txt:
        try:
            generation_len = int(generation_len_txt)
        except Exception:
            generation_len = 0
        return generation_len
    else:
        return 0


st.title('Text summary generation')
st.markdown('<span style="color: red; font-weight: bold">NB!</span> English only supported', unsafe_allow_html=True)
txt = load_text()
if len(txt) < 10:
    st.write('<span style="color: red">Your text for generation is too short ! (minimum - 10 characters)</span>',
             unsafe_allow_html=True)
txt_len = load_len()
if txt_len <= 8:
    st.write('<span style="color: red">Provide max length more than 8!</span>',
             unsafe_allow_html=True)

result = st.button('Get summary')

if result:
    try:
        with st.spinner('Generation in progress...'):
            summary = utils.generate_text(txt, txt_len)
            st.write('**Results:**')
            st.write(summary)

    except Exception as e:
        st.write(f'<span style="color: red">{e}</span>',
                 unsafe_allow_html=True)
