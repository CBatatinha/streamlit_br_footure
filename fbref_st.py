import pandas as pd
import numpy as np
import seaborn as sns
import json
from pandas.io.json import json_normalize
import math
from math import hypot
import requests
from mplsoccer.pitch import Pitch,VerticalPitch
from matplotlib.patches import Arc
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import paired_euclidean_distances
import numpy as np
from sklearn.preprocessing import StandardScaler
#### KMEANS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import gdown
import base64
import os
import ast
from ast import literal_eval
import streamlit as st
# import sys
# sys.path.append(".")
# from fbref_chute import FbrefChute

def carry_func(df):
    min_dribble_length: float = 1.0
    max_dribble_length: float = 100.0
    max_dribble_duration: float = 20.0

    def _add_dribbles(actions):
        next_actions = actions.shift(-1)
        same_team = actions.teamId == next_actions.teamId
        dx = actions.endX - next_actions.x
        dy = actions.endY - next_actions.y
        far_enough = dx ** 2 + dy **2 >= min_dribble_length **2
        not_too_far = dx ** 2 + dy **2 <= max_dribble_length **2
        dt = next_actions.time_seconds - actions.time_seconds
        same_phase = dt < max_dribble_duration
        same_period = actions.period_value == next_actions.period_value
        dribble_idx = same_team & far_enough & not_too_far & same_phase & same_period

        dribbles = pd.DataFrame()
        prev = actions[dribble_idx]
        nex = next_actions[dribble_idx]
        dribbles['game_id'] = nex.game_id
        dribbles['season_id'] = nex.season_id
        dribbles['competition_id'] = nex.competition_id
        dribbles['period_value']=nex.period_value
        for cols in ['expandedMinute']:
            dribbles[cols] = nex[cols]
        for cols in ['KP','Assist','TB']:
            dribbles[cols] = [0 for _ in range(len(dribbles))]
        dribbles['isTouch'] = [True for _ in range(len(dribbles))]
        morecols = ['position','shirtNo','playerId','hometeamid','awayteamid','hometeam','awayteam']
        for cols in morecols:
          dribbles[cols] = nex[cols]
        dribbles['actiond_id']= prev.action_id + 0.1
        dribbles['time_seconds'] = (prev.time_seconds + nex.time_seconds) / 2
        dribbles['teamId'] = nex.teamId
        dribbles['playerId'] = nex.playerId
        dribbles['name'] = nex.name
        dribbles['receiver'] = [' ' for _ in range(len(dribbles))]
        dribbles['x'] = prev.endX
        dribbles['y'] = prev.endY
        dribbles['endX'] = nex.x
        dribbles['endY'] = nex.y
        dribbles['bodypart'] = ['foot' for _ in range(len(dribbles))]
        dribbles['events'] = ['Carry' for _ in range(len(dribbles))]
        dribbles['outcome'] = ['Successful' for _ in range(len(dribbles))]
        dribbles['type_displayName'] = ['Carry' for _ in range(len(dribbles))]
        dribbles['outcome_displayName'] = ['Successful' for _ in range(len(dribbles))]
        dribbles['quals'] = [{} for _ in range(len(dribbles))]
        actions = pd.concat([actions,dribbles], ignore_index=True, sort=False)
        actions = actions.sort_values(['period_value','action_id']).reset_index(drop=True)
        actions['action_id'] = range(len(actions))
        return actions
    gamedf=df
    df['name'] = gamedf['name'].fillna(value='')
    df['action_id'] = range(len(gamedf))
    df.loc[df.type_displayName=='BallRecovery','events'] == 'NonAction'
    gameactions = ( df[df.events != 'NonAction'].sort_values(['game_id','period_value','time_seconds']).reset_index(drop=True))
    gameactions = _add_dribbles(gameactions)
    gameactions['distance']=((gameactions['endX'] - gameactions['x'])**2 + (gameactions['endY'] - gameactions['y'])**2)**0.5
    gameactions['events'].unique()
    gameactions['team']=np.where((gameactions['hometeamid']==gameactions['teamId']), gameactions.hometeam,  gameactions.awayteam)
    gameactions=gameactions.sort_values(by='time_seconds').reset_index(drop=True)
    return gameactions
def minutes_played(df):
    match=df
    match['quals'] = match.quals.apply(lambda x:ast.literal_eval(x))
    match['redcard'] = match.quals.apply(lambda x:int(33 in x or 32 in x))
    match['team']=np.where((match['hometeamid']==match['teamId']), match.hometeam,  match.awayteam)
    max_minute=match.expandedMinute.max()
    subs=match[match.type_displayName.isin(['SubstitutionOff', 'SubstitutionOn'])|
           match.redcard==1][['name','expandedMinute',
                               'team','type_displayName']].reset_index(drop=True)
    auxiliar=match.groupby('name').apply(lambda x: x['team'].unique()).apply(pd.Series).reset_index()
    auxiliar.rename(columns={0:'team'},inplace=True)
    joined=subs.append(auxiliar).fillna(0).reset_index(drop=True)
    if max_minute<105:
        joined['time_played']=np.where(joined['type_displayName']!='SubstitutionOff',90-joined['expandedMinute'],joined['expandedMinute'])
    else:
        joined['time_played']=np.where(joined['type_displayName']!='SubstitutionOff',120-joined['expandedMinute'],joined['expandedMinute'])
    joined=joined.drop_duplicates(subset='name', keep="first")
    minutes_total=joined.drop('type_displayName',axis=1)
    return minutes_total
def tabela_minute(df):
    df['team']=np.where((df['hometeamid']==df['teamId']), df.hometeam,  df.awayteam)
    li=[]
    matches=list(df.game_id.unique())
    for item in (matches):
        match=df[df['game_id']==item].reset_index(drop=True)
        minutes_total=minutes_played(match)
        li.append(minutes_total)
    frame = pd.concat(li, axis=0, ignore_index=True)
    df_minute=frame.groupby(['name','team'])['time_played'].sum().reset_index()
    df_minute=df_minute[df_minute['name']!=0].reset_index(drop=True)
    df_minute=df_minute.rename(columns={'time_played':'minutos'})
    df_minute['90']=round(df_minute['minutos']/90,2)
    return df_minute
def tabela_opp_touch(df):
    df['home_opp_touch']=np.where(((df['isTouch']==True)&(df['team']==df['awayteam'])), 1,  0)
    df['away_opp_touch']=np.where(((df['isTouch']==True)&(df['team']==df['hometeam'])), 1,  0)
    df['opp_touch']=df['home_opp_touch']+df['away_opp_touch']
    h_opp=df.groupby(['hometeam'])['home_opp_touch'].sum().reset_index().rename(columns={'hometeam':'team'})
    a_opp=df.groupby(['awayteam'])['away_opp_touch'].sum().reset_index().rename(columns={'awayteam':'team'})
    opp_touch=h_opp.merge(a_opp,how='outer', on=['team'])
    opp_touch['total_opp_touch']=opp_touch['home_opp_touch']+opp_touch['away_opp_touch']
    return opp_touch

#Shots
def tabela_chute(df):
    df_minute=tabela_minute(df)
    df['team']=np.where((df['hometeamid']==df['teamId']), df.hometeam,  df.awayteam)
    df['distance']=((df['endX'] - df['x'])**2 + (df['endY'] - df['y'])**2)**0.5
    gols=df[df['type_displayName']=='Goal'].reset_index(drop=True)
    chutes=df[df['events']=='Shot'].reset_index(drop=True)
    chutes_no_alvo=chutes[(chutes['type_displayName'].isin(['SavedShot','ShotOnPost','Goal']))]
    faltas_diretas=df[(df['events']=='Freekick')].reset_index(drop=True)
    penaltis_feitos=df[(df['events']=='Penalty')&(df['type_displayName']=='Goal')].reset_index(drop=True)
    penaltis_tentados=df[(df['events']=='Penalty')&(df['type_displayName']!='Goal')].reset_index(drop=True)
    lista_data=[gols,chutes,chutes_no_alvo,faltas_diretas,penaltis_feitos,penaltis_tentados]
    df_shot=df_minute
    for df_ in lista_data:
        df_=df_.groupby(['name','team'])['events'].count().reset_index()
        df_shot = df_shot.merge(df_,how='outer', on=['name','team'])
    lista_titulos=['gols','chutes','chutes_no_alvo','faltas_diretas','penaltis_feitos','penaltis_tentados']
    lista_titulos=[item.replace('_',' ') for item in lista_titulos]
    first_name=list(df_shot.columns[0:4])
    first_name.extend(lista_titulos)
    df_shot=df_shot.set_axis(first_name,axis='columns')
    df_shot=df_shot.fillna(0)
    try:
        df_shot['chutes no alvo %']= df_shot['chutes no alvo']/df_shot['chutes']
        df_shot['chutes no alvo %']=round(df_shot['chutes no alvo %']*100,2)
    except:
        df_shot['chutes no alvo %']=0
    df_shot=df_shot.fillna(0)
    return df_shot
#---------------------------------------------------------------
#Passes
def tabela_passes(df):
    df_minute=tabela_minute(df)
    df['team']=np.where((df['hometeamid']==df['teamId']), df.hometeam,  df.awayteam)
    df['distance']=((df['endX'] - df['x'])**2 + (df['endY'] - df['y'])**2)**0.5
    passes_certos=df[(df['type_displayName']=='Pass')&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    passes=df[(df['type_displayName']=='Pass')].reset_index(drop=True)
    passes_curtos=passes.query('distance<=15')
    passes_medios=passes.query('(distance>15)&(distance<=30)')
    passes_longos=passes.query('distance>30')
    passes_curtos_certos=passes_certos.query('distance<=15')
    passes_medios_certos=passes_certos.query('(distance>15)&(distance<=30)')
    passes_longos_certos=passes_certos.query('distance>30')
    set_piece=['freekick_short','freekick_crossed','corner_short','corner_crossed']
    bola_parada=df[df['events'].isin(set_piece)&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    infiltrado=df[(df['type_displayName']=='Pass')&(df['TB']==1)&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    passes_open_play=df[(df['type_displayName']=='Pass')&(df['events']=='Pass')&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    inversoes=df.query("(type_displayName == 'Pass')&((endY-y)**2>=36.57**2)&(outcomeType_displayName=='Successful')").reset_index(drop=True)
    cruzamentos=df[(df['events']=='cross')&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    assistencia=df[(df['type_displayName']=='Pass')&(df['Assist']==1)].reset_index(drop=True)
    passes_chave=df[(df['type_displayName']=='Pass')&(df['KP']==1)].reset_index(drop=True)
    passes_para_terço_final=passes.query("(endX>70)&(outcomeType_displayName=='Successful')").reset_index(drop=True)
    passes['dist1'] = np.sqrt((105-passes.x)**2 + (34-passes.y)**2)
    passes['dist2'] = np.sqrt((105-passes.endX)**2 + (34-passes.endY)**2)
    passes['distdiff'] = passes['dist1'] - passes['dist2']
    pass1 = passes.query("(x<52.5)&(endX<52.5)&(distdiff>=30)")
    pass2 = passes.query("(x<52.5)&(endX>52.5)&(distdiff>=15)")
    pass3 = passes.query("(x>52.5)&(endX>52.5)&(distdiff>=10)")
    pass1 = pass1.append(pass2)
    pass1 = pass1.append(pass3)
    progressivos=pass1[(pass1['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    lista_base=['Pass','cross']
    passdf = df[(df.events.isin(lista_base))&
       (df.outcomeType_displayName=='Successful')].reset_index(drop=True)
    insideboxendx = passdf.endX>=88.5
    insideboxstartx = passdf.x>=88.5
    outsideboxstartx = passdf.x<88.5
    insideboxendy = np.abs(34-passdf.endY)<20.16
    insideboxstarty = np.abs(34-passdf.y)<20.16
    allbox = passdf[~(insideboxstartx&insideboxstarty)]
    passes_para_area = allbox[insideboxendx&insideboxendy]
    lista_data=[passes_certos,passes,passes_curtos_certos,passes_curtos,
                passes_medios_certos,passes_medios,passes_longos_certos,passes_longos,
                bola_parada,infiltrado,passes_open_play,inversoes,cruzamentos,
                assistencia,passes_chave,passes_para_terço_final,progressivos,passes_para_area]
    df_passe=df_minute
    for df_ in lista_data:
        df_=df_.groupby(['name','team'])['events'].count().reset_index()
        df_passe = df_passe.merge(df_,how='outer', on=['name','team'])
    lista_titulos=['passes_certos','passes','passes_curtos_certos','passes_curtos',
                'passes_medios_certos','passes_medios','passes_longos_certos','passes_longos',
                'bola_parada','infiltrado','passes_open_play','inversoes','cruzamentos',
                'assistencia','passes_chave','passes_para_terço_final','progressivos','passes_para_area']
    lista_titulos=[item.replace('_',' ') for item in lista_titulos]
    first_name=list(df_passe.columns[0:4])
    first_name.extend(lista_titulos)
    df_passe=df_passe.set_axis(first_name,axis='columns')
    df_passe=df_passe.fillna(0)
    distancia_certo=passes_certos.groupby(['name','team'])['distance'].sum().reset_index()
    distancia_progressivos=progressivos.groupby(['name','team'])['distance'].sum().reset_index()
    df_passe['passes certos distância']=distancia_certo['distance']
    df_passe['passes progressivos distância']=distancia_progressivos['distance']
    df_passe['passes %']=round((df_passe['passes certos']/df_passe['passes'])*100,2)
    df_passe['passes curtos %']=round((df_passe['passes curtos certos']/df_passe['passes curtos'])*100,2)
    df_passe['passes medios %']=round((df_passe['passes medios certos']/df_passe['passes medios'])*100,2)
    df_passe['passes longos %']=round((df_passe['passes longos certos']/df_passe['passes longos'])*100,2)
    return df_passe
#--------------------------------------------------------------------------------------
#Defesa
def tabela_defesa(df):
    df_minute=tabela_minute(df)
    desarmes=df[(df['type_displayName']=='Tackle')].reset_index(drop=True)
    desarmes_ganhos=df[(df['type_displayName']=='Tackle')&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    desarmes_defesa=desarmes.query('x<=35').reset_index(drop=True)
    desarmes_meio=desarmes.query('(x>35)&(x<=70)').reset_index(drop=True)
    desarmes_ataque=desarmes.query('(x>70)').reset_index(drop=True)
    df['prev_action']=df['type_displayName'].shift(1)
    df['prev_outcome']=df['outcomeType_displayName'].shift(1)
    desarmes_vs_dribles=df[(df['type_displayName']=='Tackle')&(df['prev_action']=='TakeOn')].reset_index(drop=True)
    # desarmes_vs_dribles_ganhos=desarmes_vs_dribles.query("prev_outcome=='Unsuccessful'").reset_index(drop=True)
    interceptações=df[(df['type_displayName']=='Interception')&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    bloqueios=df[(df['type_displayName']=='BlockedPass')&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    cortes=df[(df['type_displayName']=='Clearance')&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    duelos_aereos_ganho=df[(df['type_displayName']=='Aerial')&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    duelos_aereos_total=df[(df['type_displayName']=='Aerial')].reset_index(drop=True)
    lista_data=[desarmes,desarmes_ganhos,desarmes_defesa,desarmes_meio,
                desarmes_ataque,desarmes_vs_dribles,
                interceptações,bloqueios,cortes,duelos_aereos_ganho,duelos_aereos_total]
    df_defesa=df_minute
    for df_ in lista_data:
        df_=df_.groupby(['name','team'])['events'].count().reset_index()
        df_defesa = df_defesa.merge(df_,how='outer', on=['name','team'])
    lista_titulos=['desarmes','desarmes_ganhos','desarmes_defesa','desarmes_meio',
                'desarmes_ataque','desarmes_vs_dribles',
                'interceptações','bloqueios','cortes','duelos_aereos_ganho','duelos_aereos_total']
    lista_titulos=[item.replace('_',' ') for item in lista_titulos]
    first_name=list(df_defesa.columns[0:4])
    first_name.extend(lista_titulos)
    df_defesa=df_defesa.set_axis(first_name,axis='columns')
    df_defesa=df_defesa.fillna(0)
    aux=tabela_opp_touch(df)
    dict_opp=aux.set_index('team').to_dict()['total_opp_touch']
    df_defesa['toques oponentes']=df_defesa['team'].map(teams_dict)
    df_defesa['adj desarmes ganhos']=round(df_defesa['desarmes_ganhos']/(df_defesa['toques oponentes']/1000),3)
    df_defesa['adj interceptações']=round(df_defesa['interceptações']/(df_defesa['toques oponentes']/1000),3)
    df_defesa['adj cortes']=round(df_defesa['cortes']/(df_defesa['toques oponentes']/1000),3)
    df_defesa['adj bloqueios']=round(df_defesa['bloqueios']/(df_defesa['toques oponentes']/1000),3)
    df_defesa['adj duelos aereos ganhos']=round(df_defesa['duelos_aereos_ganho']/(df_defesa['toques oponentes']/1000),3)
    return df_defesa
#-----------------------------------------------------------------------------
#Posse
def tabela_posse(df):
    df_minute=tabela_minute(df)
    toques=df[df['isTouch']==True].reset_index(drop=True)
    toques_defesa=toques.query('x<=35').reset_index(drop=True)
    toques_meio=toques.query('(x>35)&(x<=70)').reset_index(drop=True)
    toques_ataque=toques.query('(x>70)').reset_index(drop=True)
    dribles=df[(df['type_displayName']=='TakeOn')].reset_index(drop=True)
    dribles_certos=df[(df['type_displayName']=='TakeOn')&(df['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    li=[]
    matches=list(df.game_id.unique())
    for item in (matches):
        match=df[df['game_id']==item].reset_index(drop=True)
        carry_total=carry_func(match)
        li.append(carry_total)
    carry_df = pd.concat(li, axis=0, ignore_index=True)
    conduções=carry_df[(carry_df['events']=='Carry')&(carry_df['distance']>=5)].reset_index(drop=True)
    descarte=conduções[(conduções['endX']<conduções['x'])&(conduções['distance']>=50)].index
    conduções=conduções.drop(descarte)
    conduções_terço_final=conduções.query("(endX>70)").reset_index(drop=True)
    conduções['dist1'] = np.sqrt((105-conduções.x)**2 + (34-conduções.y)**2)
    conduções['dist2'] = np.sqrt((105-conduções.endX)**2 + (34-conduções.endY)**2)
    conduções['distdiff'] = conduções['dist1'] - conduções['dist2']
    prog1 = conduções.query("(x<52.5)&(endX<52.5)&(distdiff>=30)")
    prog2 = conduções.query("(x<52.5)&(endX>52.5)&(distdiff>=15)")
    prog3 = conduções.query("(x>52.5)&(endX>52.5)&(distdiff>=10)")
    prog1 = prog1.append(prog2)
    prog1 = prog1.append(prog3)
    conduções_progressivas=prog1.reset_index(drop=True)
    insideboxendx = conduções.endX>=88.5
    insideboxstartx = conduções.x>=88.5
    outsideboxstartx = conduções.x<88.5
    insideboxendy = np.abs(34-conduções.endY)<20.16
    insideboxstarty = np.abs(34-conduções.y)<20.16
    allbox = conduções[~(insideboxstartx&insideboxstarty)]
    conduções_para_area = allbox[insideboxendx&insideboxendy]
    lista_data=[toques,toques_defesa,toques_meio,toques_ataque,
                dribles_certos,dribles,conduções,conduções_terço_final,
                conduções_para_area,conduções_progressivas]
    df_posse=df_minute
    for df_ in lista_data:
        df_=df_.groupby(['name','team'])['events'].count().reset_index()
        df_posse = df_posse.merge(df_,how='outer', on=['name','team'])
    lista_titulos=['toques','toques_defesa','toques_meio','toques_ataque',
                'dribles_certos','dribles','conduções','conduções_terço_final',
                'conduções_para_area','conduções_progressivas']
    lista_titulos=[item.replace('_',' ') for item in lista_titulos]
    first_name=list(df_posse.columns[0:4])
    first_name.extend(lista_titulos)
    df_posse=df_posse.set_axis(first_name,axis='columns')
    df_posse=df_posse.fillna(0)
    # distancia_total=conduções.groupby(['name','team'])['distance'].sum().reset_index()
    # distancia_progressivas=conduções_progressivas.groupby(['name','team'])['distance'].sum().reset_index()
    # df_posse['conduções distância total']=distancia_total['distance']
    # df_posse['conduções progressivas distância']=distancia_progressivas['distance']
    df_posse.fillna(0)
    return df_posse






#-------------------
