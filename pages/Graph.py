import time
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from BernoulliNaiveBayes import *
from ComplementNaiveBayes import precision
from StressMain import ytest
from Home import user_input

st.set_page_config(
    page_title="Accuracy Graphs",
    page_icon="📈",
)

st.title("Accuracy Graphs")

st.sidebar.header("Graphs")


scalex = [10]
y = precision(ytest, user_input)
plt.plot(scalex, y)