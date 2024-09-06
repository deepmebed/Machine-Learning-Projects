import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the dataset with header=1
file_path = 'Winner Team prediction  dataset.xlsx'

@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path, sheet_name='Sheet1', header=1)
    return data

data = load_data(file_path)

image_path = 'iub-the-islamia-university-of-bahawalpur3406.jpg'
st.image(image_path,width=300)

st.title("T20 2024 Winner Team Prediction Analysis")

# Sidebar for filtering data
st.sidebar.title("Filters")

# year filter
year_options = data['Year'].unique()
year_filter = st.sidebar.selectbox("Year", options=['Select a year'] + [int(year) for year in year_options if not pd.isna(year)])



# Filter teams based on selected year
if year_filter != 'Select a year':
    team_options = data[data['Year'] == year_filter]['Team1'].unique()
    team1_filter = st.sidebar.selectbox("Team 1", options=['Select a team'] + list(team_options))

    if team1_filter != 'Select a team':
        team2_options = data[(data['Year'] == year_filter) & (data['Team1'] == team1_filter)]['Team2'].unique()
        team2_filter = st.sidebar.selectbox("Team 2 (Opponent)", options=['Select an opponent'] + list(team2_options))
    else:
        team2_filter = 'Select an opponent'
else:
    team1_filter = 'Select a team'
    team2_filter = 'Select an opponent'

# Apply filters if both teams and year are selected
if team1_filter != 'Select a team' and team2_filter != 'Select an opponent' and year_filter != 'Select a year':
    filtered_data = data[(data['Team1'] == team1_filter) & (data['Team2'] == team2_filter) & (data['Year'] == year_filter)]

    
    if not filtered_data.empty:
        winner_team_input = filtered_data['WinnerTeam'].iloc[0]
        mtm_input = filtered_data['MTM'].iloc[0]
        toss_decision_input = filtered_data['TossDecisiion'].iloc[0]
    else:
        winner_team_input = ''
        mtm_input = ''
        toss_decision_input = ''

   
    st.text_input("Winner Team", value=winner_team_input, key='winner_team_input')
    st.text_input("Man of the Match", value=mtm_input, key='mtm_input')
    st.text_input("Toss Decision", value=toss_decision_input, key='toss_decision_input')

  
    st.subheader("Insights")

  
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Number of Wins by Team", "Man of the Match Awards", "Impact Player Awards"))

    # Winner Team Distribution
    winner_team_count = filtered_data['WinnerTeam'].value_counts()
    fig.add_trace(go.Bar(x=winner_team_count.index, y=winner_team_count.values, name='Wins'), row=1, col=1)

    # Man of the Match
    mtm_count = filtered_data['MTM'].value_counts()
    fig.add_trace(go.Bar(x=mtm_count.index, y=mtm_count.values, name='Man of the Match'), row=1, col=2)

    # Impact Player
    if 'ImpactPlayer' in filtered_data.columns:
        impact_player_count = filtered_data['ImpactPlayer'].value_counts()
        fig.add_trace(go.Bar(x=impact_player_count.index, y=impact_player_count.values, name='Impact Player'), row=2, col=1)

    
    fig.update_layout(height=800, width=800, title_text="Insights of Selected Teams", showlegend=False)
    st.plotly_chart(fig)

    # Performance of Winner Team in the Selected Year
    if winner_team_input:
        total_matches = data[(data['Year'] == year_filter) & ((data['Team1'] == winner_team_input) | (data['Team2'] == winner_team_input))].shape[0]
        win_matches = data[(data['Year'] == year_filter) & (data['WinnerTeam'] == winner_team_input)].shape[0]
        
        st.subheader(f"Performance of {winner_team_input} in {year_filter}")
        st.write(f"Total Matches: {total_matches}")
        st.write(f"Win Matches: {win_matches}")

        performance_fig = go.Figure(data=[
            go.Bar(name='Total Matches', x=[year_filter], y=[total_matches]),
            go.Bar(name='Win Matches', x=[year_filter], y=[win_matches])
        ])
        
        performance_fig.update_layout(barmode='group', title_text=f"Total Matches vs Win Matches for {winner_team_input} in {year_filter}")
        st.plotly_chart(performance_fig)

    # Performance Comparison with Other Teams
    st.subheader(f"Performance Comparison in {year_filter}")
    
    
    teams_in_year = data[data['Year'] == year_filter]['Team1'].unique()
    
    comparison_data = []
    for team in teams_in_year:
        total_matches_team = data[(data['Year'] == year_filter) & ((data['Team1'] == team) | (data['Team2'] == team))].shape[0]
        win_matches_team = data[(data['Year'] == year_filter) & (data['WinnerTeam'] == team)].shape[0]
        comparison_data.append({
            'Team': team,
            'Total Matches': total_matches_team,
            'Win Matches': win_matches_team
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    comparison_fig = go.Figure(data=[
        go.Bar(name='Total Matches', x=comparison_df['Team'], y=comparison_df['Total Matches']),
        go.Bar(name='Win Matches', x=comparison_df['Team'], y=comparison_df['Win Matches'])
    ])
    
    comparison_fig.update_layout(barmode='group', title_text=f"Total Matches vs Win Matches for All Teams in {year_filter}")
    st.plotly_chart(comparison_fig)

    # Performance Indicator: Comparing current year with previous year
    st.subheader("Performance Indicator")

    previous_year = year_filter - 1
    if previous_year in year_options:
        previous_year_data = data[data['Year'] == previous_year]
        current_year_data = data[data['Year'] == year_filter]

        prev_year_wins = previous_year_data[previous_year_data['WinnerTeam'] == winner_team_input].shape[0]
        curr_year_wins = current_year_data[current_year_data['WinnerTeam'] == winner_team_input].shape[0]

        if curr_year_wins > prev_year_wins:
            performance_indicator = "Performance is improving"
        elif curr_year_wins < prev_year_wins:
            performance_indicator = "Performance is declining"
        else:
            performance_indicator = "Performance remains the same"

        st.write(f"Comparison with previous year ({previous_year}): {performance_indicator}")
        st.write(f"Wins in {previous_year}: {prev_year_wins}")
        st.write(f"Wins in {year_filter}: {curr_year_wins}")

    else:
        st.write(f"No data available for the previous year ({previous_year}) for comparison.")

else:
    st.write("Please select Year, Team 1, and Team 2 to view insights.")
