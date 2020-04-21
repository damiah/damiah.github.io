import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime

#Read in player logs
hist_player_log_full = pd.read_csv('hist_player_log.csv')
#Read in player injury history
full_injury_data = pd.read_csv('full_injury_data_clean.csv', parse_dates=['Date'])
#Read in player info
hist_player_info = pd.read_csv('hist_player_info.csv', parse_dates=['BIRTHDATE'])

#slice player log
hist_player_log_slice = hist_player_log_full[['full_name', 'GAME_DATE']]
hist_player_log_slice['Date'] = hist_player_log_slice['GAME_DATE'].apply(lambda x: datetime.strptime(x, '%b %d, %Y'))
hist_player_log_slice['InjuryFlag'] = 0
hist_player_log_slice['GameFlag'] = 1


#slice injury data
full_injury_data_slice = full_injury_data[['full_name_main', 'Date', 'injury_anat_partone']]
full_injury_data_slice['Date'] = pd.to_datetime(full_injury_data_slice['Date'])
full_injury_data_slice['InjuryFlag'] = 1
full_injury_data_slice['GameFlag'] = 0


#concat games df with injury df
full_injury_game_slice = pd.concat([full_injury_data_slice.rename(columns={'full_name_main':'full_name'}),
                                    hist_player_log_slice[['full_name', 'Date', 'InjuryFlag','GameFlag']]],ignore_index=True)
full_injury_game_slice = full_injury_game_slice.sort_values(['full_name', 'Date'], ascending=[1,1])
full_injury_game_slice['injury_anat_partone'] = full_injury_game_slice['injury_anat_partone'].fillna('')
full_injury_game_slice = full_injury_game_slice.groupby(['full_name', 'Date']).agg({'GameFlag' : 'sum',
                                                          'InjuryFlag' : 'sum',
                                                          'injury_anat_partone' : 'sum'}).reset_index()


#Identify the instances where the injury turns up twice in a row (without game in between) for a player
full_injury_game_slice['future_inj'] = full_injury_game_slice['InjuryFlag'].shift(-1)
full_injury_game_slice['future_inj_anat'] = full_injury_game_slice['injury_anat_partone'].shift(-1)
full_injury_game_slice['future_full_name'] = full_injury_game_slice['full_name'].shift(-1)
injury_dupes = full_injury_game_slice[((full_injury_game_slice['future_inj'] == full_injury_game_slice['InjuryFlag']) 
     & (full_injury_game_slice['future_inj_anat'] == full_injury_game_slice['injury_anat_partone'])
     & (full_injury_game_slice['full_name'] == full_injury_game_slice['future_full_name'])
     &(full_injury_game_slice['future_inj_anat'] != '') )]

#Merge and eliminate
full_injury_data_final = full_injury_data.merge(injury_dupes[['full_name','Date']].drop_duplicates(), left_on=['full_name_main','Date']
                                , right_on=['full_name','Date'], 
                   how='left', indicator=True)
full_injury_data_final = full_injury_data_final[full_injury_data_final['_merge']=='left_only']
full_injury_data_final = full_injury_data_final.merge(hist_player_info, left_on='full_name_main',right_on='full_name',how='inner')
full_injury_data_final[full_injury_data_final['full_name_main']=='Kawhi Leonard'].head()