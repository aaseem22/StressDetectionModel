import streamlit as st
import pandas as pd
from StressMain import NbMod, NbMod2

st.title("Stress Detection")
st.write("""
# First APP
Hello *Salman*
""")

txt1 = st.text_input('Enter Your Sentence', placeholder='Start Typing Here')
output4 = NbMod2(txt1)
st.write(output4)

