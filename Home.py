import streamlit as st
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from ComplementNaiveBayes import *

from StressMain import NbMod2, ytest, DecisionTreeClassifier, xtrain, ytrain, DecisionTree, DecisionTreeAcc, Logistic, \
    LogisticAcc


st.set_page_config(
    page_title="Stress Detection",
    page_icon="😵"
)

st.title("Stress Detection")

st.sidebar.header('Home')

user_input: str = st.text_area('Enter Sentence To Be Analyzed ', placeholder='Start Typing...',value=".")

xz = user_input
#
# nb_acc = calculate_accuracy(model, xtest, ytest)
# lr_acc = calculate_accuracy(LogisticRegression11(), xtest, ytest)







def button():




        # st.title("Bernoulli Naive Bayes")
        # output2 = NbMod(user_input)
        # st.write(output2)

        # st.title("Compliment Naive Bayes")
        # output4 = NbMod2([user_input])
        # st.write(output4)
        #
        st.title("Logistic Regression")
        output4 = Logistic(user_input)
        st.write(output4)
        acc_lr = LogisticAcc(user_input)


        # acc1 = accuracy(user_input)
        # st.write(acc1)
        st.title("Decision Tree")
        outputdt = DecisionTree(user_input)
        st.write(outputdt)




if st.button("Analyze"):
    button()
#
# if st.button("Analyse"):
#     # st.title("Bernoulli Naive Bayes")
#     # output2 = NbMod(user_input)
#     # st.write(output2)
#
#     st.title("Compliment Naive Bayes")
#     output4 = NbMod2([user_input])
#     st.write(output4)
#
#     # acc1 = accuracy(user_input)
#     # st.write(acc1)
#     st.title("Decision Tree")
#     output4 = DT(user_input)
#     st.write(output4)
#     st.write(accuracy(user_input,output4))
#
#
#
#
#
