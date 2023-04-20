import streamlit as st

from Home import user_input

st.set_page_config(
    page_title="Accuracy Graphs",
    page_icon="📈",
)


from plotly.express import data, line
from StressMain import LogisticAcc, DecisionTreeAcc, CNBAccuracy



st.title("Accuracy Graphs")

st.sidebar.header("Graphs")

acc_lr = LogisticAcc(user_input)*100
acc_dt = DecisionTreeAcc(user_input)*100
acc_cn = CNBAccuracy(user_input)*100
models = ['Logistic Regression', 'Decision Tree','Compliment Naive Bayes']
accuracies = [acc_lr, acc_dt,acc_cn]

df = ({"Model": models, "Accuracy in %": accuracies})

fig = line(df, x="Model", y="Accuracy in %")

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit")
with tab2:
    st.plotly_chart(fig, theme=None)

