
import streamlit as st
from Home import user_input
from StressMain import LogisticAcc, DecisionTreeAcc

st.set_page_config(
    page_title="Accuracy Graphs",
    page_icon="📈",
)

st.title("Accuracy Graphs")

st.sidebar.header("Graphs")

acc_lr = LogisticAcc(user_input)
acc_dt = DecisionTreeAcc(user_input)
models = ['Logistic Regression', 'Decision Tree', ]
accuracies = [acc_lr, acc_dt]
@st.cache_data
import plotly.express as px
df = px.data.gapminder().query("continent == 'Oceania'")
fig = px.line(df, x='Models', y='Accuracy', color='country', symbol="country")

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit")
with tab2:
    st.plotly_chart(fig, theme=None)