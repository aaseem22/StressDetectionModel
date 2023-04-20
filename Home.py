import streamlit as st

st.set_page_config(
    page_title="Stress Detection",
    page_icon="😵"
)

from StressMain import DecisionTree, ComplimentNaiveBayes11, Logistic


st.title("Stress Detection")

st.sidebar.header('Home')

user_input: str = st.text_area('Enter Your :red[Thoughts] To Be Analyzed ', placeholder='Start Typing...',value=".")

st.markdown('Express as much as possible for better :blue[accuracy]  :heart:')

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