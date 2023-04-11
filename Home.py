import easygui as easygui
import streamlit as st

from StressMain import NbMod, NbMod2

st.set_page_config(
    page_title="Stress Detection",
    page_icon="😵"
)




st.title("Stress Detection")

st.sidebar.header('Home')

user_input = st.text_area('Enter Sentence To Be Analyzed ', placeholder='Start Typing...')

if st.button("Analyze"):
    st.title("Bernoulli Naive Bayes")
    output2 = NbMod(user_input)
    st.write(output2)

    st.title("Compliment Naive Bayes")
    output4 = NbMod2(user_input)
    st.write(output4)
