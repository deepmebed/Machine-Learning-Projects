import numpy as np
import pandas as pd
import pickle
import streamlit as st

pipe = pickle.load(open('pipe.pkl','rb'))

teams = ['Australia',
 'India',
 'Bangladesh',
 'New Zealand',
 'South Africa',
 'England',
 'West Indies',
 'Afghanistan',
 'Pakistan',
 'Sri Lanka']

cities = ['Colombo',
 'Mirpur',
 'Johannesburg',
 'Dubai',
 'Auckland',
 'Cape Town',
 'London',
 'Pallekele',
 'Barbados',
 'Sydney',
 'Melbourne',
 'Durban',
 'St Lucia',
 'Wellington',
 'Lauderhill',
 'Hamilton',
 'Centurion',
 'Abu Dhabi',
 'Manchester',
 'Mumbai',
 'Nottingham',
 'Southampton',
 'Mount Maunganui',
 'Chittagong',
 'Kolkata',
 'Lahore',
 'Delhi',
 'Nagpur',
 'Chandigarh',
 'Adelaide',
 'Bangalore',
 'St Kitts',
 'Cardiff',
 'Christchurch',
 'Trinidad']

st.title('Cricket score prediction')

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('select batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('select bowling team',sorted(teams))

city = st.selectbox('select city',sorted(cities))

col3,col4,col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current score')
with col4:
    over = st.number_input('overs Done(work for over > 5)')
with col5:
    wickets = st.number_input('wickets out')
last_five = st.number_input('runs scored in last 5 overs')

if st.button('predict score'):
    balls_left = (120 - over * 6)
    wickets_left = 10 - wickets
    crr = current_score / over

    input_df = pd.DataFrame({
        'batting_team':[batting_team], 'bowling_team':[bowling_team], 'city':city,'current_score':[current_score],
        'balls_left':[balls_left],'wickets_left':[wickets_left],'crr':[crr],'last_five':[last_five]
        })

    try:
        result = pipe.predict(input_df)
        st.header('Predicted Score: ' + str(int(result[0])))
    except Exception as e:
        st.error(f"An error occurred: {e}")