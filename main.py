import streamlit as st
import pandas as pd


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Welcome to my awesome data science project')
    st.text('In this project, I look into transactions of taxis in NYC...')

with dataset:
    st.header('NYC taxi dataset')
    st.text('I made this up as below')
    df = pd.read_csv('C://Users//Arjun//PycharmProjects//streamlit_test_001//test_file_data_001.csv')
    st.write(df.head(7))

with features:
    st.header('The features I created')

with model_training:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparametes of the model and see how the performance changes')

