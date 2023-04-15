import streamlit as st
from StressMain import DecisionTree, ComplimentNaiveBayes11, Logistic

st.set_page_config(
    page_title="Stress Detection",
    page_icon="😵"
)

st.title("Stress Detection")

st.sidebar.header('Home')

user_input: str = st.text_area('Enter Sentence To Be Analyzed ', placeholder='Start Typing...',value=".")

xz = user_input


def button():

        st.title("Logistic Regression")
        output_lr = Logistic(user_input)
        st.write(output_lr)

        st.title("Decision Tree")
        output_dt = DecisionTree(user_input)
        st.write(output_dt)

        st.title("Compliment Naive Bayes")
        output_cnb = ComplimentNaiveBayes11(user_input)
        st.write(output_cnb)




if st.button("Analyze"):
    button()