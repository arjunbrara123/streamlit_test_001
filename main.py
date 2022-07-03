import streamlit as st
import pandas as pd
import numpy as np
import base64

header = st.container()
intro = st.container()
dataset = st.container()
factors = st.container()
model_training = st.container()

with header:
    st.title("We tried AmEx's $100k Credit Risk Modelling Challenge! 🏆")
    st.text("We didn't win the cash, but we learned a lot along the way... 🤓")
    st.markdown("![Alt Text](https://media.giphy.com/media/3o6UB5RrlQuMfZp82Y/giphy.gif)")

with intro:
    st.header('The Challenge...')
    st.text("We were given 500,000 customers sets of anonymised bank statements")
    st.text("... and if they defaulted on their credit card or not")
    st.text("which we used to build a model")
    st.text("We were then given 1,000,000 customers sets of anonymised bank statements")
    st.text("...where we didn't know if they defaulted on their credit card or not")
    st.text("In order to test our earlier built model on")
    st.text("The most accurate predictions on the 1,000,000 got the cash!")

    pc_path = 'C://Users//Arjun//PycharmProjects//streamlit_test_001//test_file_data_001.csv'
    gt_path = 'https://raw.githubusercontent.com/arjunbrara123/streamlit_test_001/master/test_file_data_001.csv'
    # df = pd.read_csv(pc_path)
    # st.write(df.head(7))

with dataset:
    st.header('The Dataset')
    st.text("We started the challenge with an actuarial lens")
    st.text("By splitting the data into similar behaviour groups we wanted to model")
    st.text("The most obvious was those with AmEx cards for less than a year, and the longer term ones")
    st.text("These had very different default rates, as shown in the graph below")
    st.text("(The dataset had max 13 statements, so 13 month and longer term ones are bundled into 13+ months)")
    month_statements, a = st.columns(2)
    set_groups = month_statements.slider('Statement Month Groups:', min_value=2, max_value=13, value=13)
    statement_group_months = list(range(1, 14))
    statement_group_default_rates = [0.3357, 0.3185, 0.3586, 0.4162, 0.3926, 0.3877, 0.4184, 0.4473, 0.4502, 0.4623,
                                     0.4467, 0.3893, 0.2318]
    bins = np.zeros(set_groups)
    bin_fac = np.zeros(set_groups)
    j = 0
    rate_num = 0
    for i in range(13):
            #st.text(i)
            #st.text(int(i*set_groups/13))
            #st.text(bins[int(i*set_groups/13)])
            bins[int(i*set_groups/13)] += (statement_group_default_rates[i])
            #st.text(bins[int(i*set_groups/13)])
            bin_fac[int(i*set_groups/13)] += 1
    bins = bins / bin_fac

    #st.write(bins)
    df_chart = pd.DataFrame(bins, range(1, set_groups+1))
    #st.write(df_chart)
    st.bar_chart(df_chart)
    st.text("After trying the different group sets you can play about with above...")
    st.text("We felt group '13' was very different, and grouped the rest into a seperate category")

with factors:
    st.header('Outliers and Obvious Factor-Based Split')
    st.text('We found certain obvious factors and trends which helped with some of the outliers')
    st.text('You can view some of these using the slider below')
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider('Outlier Threshold To View The Impact Of:', min_value=10, max_value=90, value=80)
    st.text(f"Viewing Outlier Threshold of: {max_depth}%")

with model_training:
    st.header('Introducing the AI and Deep Learning models')
    st.text(
        'Finally, having got our data into appropriate behavioural groups, it was time to start fitting deep learning neural network models to these.')
