import pandas as pd
import numpy as np
from mplsoccer.pitch import Pitch,VerticalPitch
from matplotlib.patches import Arc
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import ast
from ast import literal_eval
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import PIL
from PIL import Image
from PIL import Image, ImageDraw, ImageFilter
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageOps
from fbref_st import *
import base64
import os

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href
def gamestate(df):
    gameactions=df
    condition_h=(gameactions['type_displayName']=='Goal') & (gameactions['outcome']=='Successful') & (gameactions['hometeamid']==gameactions.teamId)
    condition_h_own=(gameactions['type_displayName']=='Goal') & (gameactions['outcome']=='OwnGoal') & (gameactions['hometeamid']!=gameactions.teamId)
    condition_a=(gameactions['type_displayName']=='Goal') & (gameactions['outcome']=='Successful') & (gameactions['awayteamid']==gameactions.teamId)
    condition_a_own=(gameactions['type_displayName']=='Goal') & (gameactions['outcome']=='OwnGoal') & (gameactions['awayteamid']!=gameactions.teamId)
    gameactions['isHomeGoal']=np.where((condition_h|condition_h_own), 1, 0)
    gameactions['isAwayGoal']=np.where((condition_a|condition_a_own), 1, 0)
    gameactions['h_goal']=gameactions['isHomeGoal'].cumsum()
    gameactions['a_goal']=gameactions['isAwayGoal'].cumsum()
    gameactions['h_saldo']=gameactions['h_goal']-gameactions['a_goal']
    gameactions['a_saldo']=gameactions['a_goal']-gameactions['h_goal']
    return gameactions
def possession_calc(Df, min1, min2):
    home = Df[(Df.teamId==Df.hometeamid)]
    away = Df[(Df.teamId==Df.awayteamid)]
    homepass = home.query("(type_displayName=='Pass')&\
                (outcomeType_displayName=='Successful')")
    awaypass = away.query("(type_displayName=='Pass')&\
                (outcomeType_displayName=='Successful')")
    homepass3 = homepass[homepass.x>70]
    awaypass3 = awaypass[awaypass.x>70]
    homepasses = len(homepass[(homepass.expandedMinute>=min1)&
                        (homepass.expandedMinute<=min2)])
    awaypasses = len(awaypass[(awaypass.expandedMinute>=min1)&
                        (awaypass.expandedMinute<=min2)])
    home3 = len(homepass3[(homepass3.expandedMinute>=min1)&
                        (homepass3.expandedMinute<=min2)])
    away3 = len(awaypass3[(awaypass3.expandedMinute>=min1)&
                        (awaypass3.expandedMinute<=min2)])
    return round(homepasses*100.0/(homepasses+awaypasses)),round(home3*100.0/(home3+away3))
def PPDAcalculator(Df,min1,min2):
  home = Df[(Df.teamId==Df.hometeamid)&(Df.expandedMinute>=min1)&
            (Df.expandedMinute<=min2)]
  away = Df[(Df.teamId==Df.awayteamid)&(Df.expandedMinute>=min1)&
             (Df.expandedMinute<=min2)]
  homedef = len(home.query("(type_displayName in ['Tackle','Interception','Challenge'])&\
                                   (x>42)"))
  homepass = len(home.query("(type_displayName=='Pass')&(x<63)&\
                   (outcomeType_displayName=='Successful')"))
  homefouls = len(home.query("(type_displayName=='Foul')&\
           (outcomeType_displayName=='Unsuccessful')&(x>42)"))
  awaydef = len(away.query("(type_displayName in ['Tackle','Interception','Challenge'])&\
                                   (x>42)"))
  awaypass = len(away.query("(type_displayName=='Pass')&(x<63)&\
                       (outcomeType_displayName=='Successful')"))
  awayfouls = len(away.query("(type_displayName=='Foul')&\
                       (outcomeType_displayName=='Unsuccessful')&(x>42)"))
  PPDAhome = round(awaypass/(homedef+homefouls)) if (homedef+homefouls)>0 else 0
  PPDAaway = round(homepass/(awaydef+awayfouls)) if (awaydef+awayfouls)>0 else 0
  return PPDAhome, PPDAaway
def shot_calculator(df):
    shots=df[df['events']=='Shot'].reset_index(drop=True)
    goal=shots[shots['type_displayName']=='Goal'].reset_index(drop=True)
    shot_off=shots[(shots['type_displayName']=='MissedShots')].reset_index(drop=True)
    shot_on=shots[(shots['type_displayName'].isin(['SavedShot','ShotOnPost']))].reset_index(drop=True)
    penalty=df[(df['events']=='Penalty')&(df['type_displayName']=='Goal')].reset_index(drop=True)
    penalty_on=df[(df['events']=='Penalty')&(df['type_displayName']=='SavedShot')].reset_index(drop=True)
    penalty_off=df[(df['events']=='Penalty')&(df['type_displayName']=='MissedShots')].reset_index(drop=True)
    fk=df[(df['events']=='Freekick')&(df['type_displayName']=='Goal')].reset_index(drop=True)
    fk_on=df[(df['events']=='Freekick')&(df['type_displayName']=='SavedShot')].reset_index(drop=True)
    fk_off=df[(df['events']=='Freekick')&(df['type_displayName']=='MissedShots')].reset_index(drop=True)
    try:
      target=len(goal)+len(penalty)+len(fk)+len(shot_on)+len(penalty_on)+len(fk_on)
    except:
      target=0
    try:
      total = len(goal)+len(penalty)+len(fk)+len(shot_on)+len(penalty_on)+len(fk_on)+len(shot_off)+len(penalty_off)+len(fk_off)
    except:
      total=0
    return (target,total)

def recover_calculator(df):
    recover = df[(df.events.isin(['BallRecovery']))&
        (df.outcomeType_displayName=='Successful')].reset_index(drop=True)
    return(len(recover))
def entry_calculator(df):
    lista_base=['Pass','cross','carry']
    lista_final=lista_base
    passdf = df[(df.events.isin(lista_final))&
       (df.outcomeType_displayName=='Successful')].reset_index(drop=True)
    insideboxendx = passdf.endX>=88.5
    insideboxstartx = passdf.x>=88.5
    outsideboxstartx = passdf.x<88.5
    insideboxendy = np.abs(34-passdf.endY)<20.16
    insideboxstarty = np.abs(34-passdf.y)<20.16
    allbox = passdf[~(insideboxstartx&insideboxstarty)]
    allbox = allbox[insideboxendx&insideboxendy]
    return (len(allbox))

xtd = np.array([[0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267,
        0.01248344, 0.01473596, 0.0174506 , 0.02122129, 0.02756312,
        0.03485072, 0.0379259 ],
       [0.00750072, 0.00878589, 0.00942382, 0.0105949 , 0.01214719,
        0.0138454 , 0.01611813, 0.01870347, 0.02401521, 0.02953272,
        0.04066992, 0.04647721],
       [0.0088799 , 0.00977745, 0.01001304, 0.01110462, 0.01269174,
        0.01429128, 0.01685596, 0.01935132, 0.0241224 , 0.02855202,
        0.05491138, 0.06442595],
       [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646,
        0.01484598, 0.01689528, 0.0199707 , 0.02385149, 0.03511326,
        0.10805102, 0.25745362],
       [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646,
        0.01484598, 0.01689528, 0.0199707 , 0.02385149, 0.03511326,
        0.10805102, 0.25745362],
       [0.0088799 , 0.00977745, 0.01001304, 0.01110462, 0.01269174,
        0.01429128, 0.01685596, 0.01935132, 0.0241224 , 0.02855202,
        0.05491138, 0.06442595],
       [0.00750072, 0.00878589, 0.00942382, 0.0105949 , 0.01214719,
        0.0138454 , 0.01611813, 0.01870347, 0.02401521, 0.02953272,
        0.04066992, 0.04647721],
       [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267,
        0.01248344, 0.01473596, 0.0174506 , 0.02122129, 0.02756312,
        0.03485072, 0.0379259 ]])

x = np.linspace(0,105,12)
y = np.linspace(0,68,8)
f = RegularGridInterpolator((y, x), xtd)
def binnings(Df,f):
    Df['start_zone_value'] = Df[['x', 'y']].apply(lambda x: f([x[1],x[0]])[0], axis=1)
    Df['end_zone_value'] = Df[['endX', 'endY']].apply(lambda x: f([x[1],x[0]])[0], axis=1)
    Df['xt_value'] = Df['end_zone_value'] - Df['start_zone_value']
    return Df
def xTplotter(Df):
    df = Df.copy()
    df_events = df.query("(events in ['Pass','cross','carry'])&\
                            (outcome in 'Successful')")
    df_xt = binnings(df_events,f).reset_index(drop=True)
    return df_xt

def xt_calculator(df):
    match=xTplotter(df)
    descarte=match[(match['endX']<match['x'])&(match['distance']>=50)].index
    match=match.drop(descarte)
    match=match[match['xt_value']>0].reset_index(drop=True)
    homemoves=match[match['hometeam']==match.team].reset_index(drop=True)
    awaymoves = match[match['awayteam']==match.team].reset_index(drop=True)
    homemoves['xt_cumu'] = homemoves.xt_value.cumsum()
    awaymoves['xt_cumu'] = awaymoves.xt_value.cumsum()
    from scipy.ndimage.filters import gaussian_filter1d
    homexTlist = [homemoves.query("(time_seconds>=300*@i)&(time_seconds<=300*(@i+1))").\
            xt_value.sum() for i in range(round(match.time_seconds.max()/60//5)+1)]
    awayxTlist = [awaymoves.query("(time_seconds>=300*@i)&(time_seconds<=300*(@i+1))").\
            xt_value.sum() for i in range(round(match.time_seconds.max()/60//5)+1)]
    homexTlist = gaussian_filter1d(homexTlist, sigma=1)
    awayxTlist = gaussian_filter1d(awayxTlist, sigma=1)
    home_xt_cum=np.cumsum(homexTlist)
    away_xt_cum=np.cumsum(awayxTlist)
    home_xt=round(home_xt_cum[-1],2)
    away_xt=round(away_xt_cum[-1],2)
    return (home_xt,away_xt)

def posse_media(df):
    maxminutes = df.expandedMinute.max()
    hp_media = int(possession_calc(df,0,maxminutes)[0])
    ap_media = int(100-possession_calc(df,0,maxminutes)[0])
    return(hp_media,ap_media)
def ft_media(df):
    maxminutes = df.expandedMinute.max()
    hft_media = int(possession_calc(df,0,maxminutes)[1])
    aft_media = int(100-possession_calc(df,0,maxminutes)[1])
    int(possession_calc(df,0,maxminutes)[0])
    return(hft_media,aft_media)
def ppda_media(df):
    maxminutes = df.expandedMinute.max()
    h_media = (PPDAcalculator(df,0,maxminutes)[0])
    a_media = int(PPDAcalculator(df,0,maxminutes)[1])
    return(h_media,a_media)

def summary_plot(df,home_team,away_team):
    df_final=carry_func(df)
    df_final=gamestate(df_final)
    home_df=df_final[df_final['hometeam']==df_final.team].reset_index(drop=True)
    away_df=df_final[df_final['awayteam']==df_final.team].reset_index(drop=True)
    bar_stats=['Gols','Finalizações','Finalizações no alvo','Posse','Dominio Territorial',
                'PPDA','Perigo gerado (xT)','Retomadas de bola','Entradas na área']
    home_stats=[df_final['h_goal'].max(),shot_calculator(home_df)[1],shot_calculator(home_df)[0],
                posse_media(df)[0],ft_media(df)[0],ppda_media(df)[0],xt_calculator(df_final)[0],
                recover_calculator(home_df),entry_calculator(home_df)]
    away_stats=[df_final['a_goal'].max(),shot_calculator(away_df)[1],shot_calculator(away_df)[0],
                posse_media(df)[1],ft_media(df)[1],ppda_media(df)[1],xt_calculator(df_final)[1],
                recover_calculator(away_df),entry_calculator(away_df)]
    total=[x + y for x, y in zip(home_stats, away_stats)]
    home_per=[(x/y)*100 for x, y in zip(home_stats, total)]
    away_per=[(x/y)*100 for x, y in zip(away_stats, total)]
    home_per[5]=away_per[5]
    away_per[5]=100-home_per[5]

    cor_fundo='#2c2b2b'
    home_color = '#00ADB9'
    away_color = '#FFA966'
    fig, ax = plt.subplots(figsize=(9,9))
    fig.set_facecolor(cor_fundo)
    ax.patch.set_facecolor(cor_fundo)

    bar_home = ax.barh(bar_stats, home_per, height=0.5,
                       alpha=1, facecolor=home_color, edgecolor='k', linewidth=3,hatch='\\')
    bar_away = ax.barh(bar_stats, away_per, height=0.5,
                       alpha=1, facecolor=away_color, edgecolor='k', linewidth=3,hatch='\\',
                       left=home_per)

    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    for i, stat in enumerate(bar_stats):
            ax.text(50, i-0.4, s=f'{stat}',ha='center', va='center',
                    fontsize=16, fontweight='bold',color='w',zorder=3)

    # Add home and away stats
    for i, home_stat in enumerate(home_stats):
        ax.text(5, i-0.4, s=f'{home_stat}', ha='center', va='center',
                fontsize=18, fontweight='bold',color='w',zorder=3)
    for i, away_stat in enumerate(away_stats):
        ax.text(95, i-0.4, s=f'{away_stat}', ha='center', va='center',
                fontsize=18, fontweight='bold',color='w',zorder=3)
    # ax.axvline(x=50, linestyle='dashed', alpha=0.5, color='#eeeeee', zorder=2)
    ax.invert_yaxis()


    # ax.tick_params(axis='y', colors='w', labelsize=14)
    plt.savefig(f'content/Report_{home_team}_{away_team}.png',dpi=300,facecolor=cor_fundo)
    im=Image.open(f'content/Report_{home_team}_{away_team}.png')
    cor_fundo = '#2c2b2b'
    tamanho_arte = (3000, 2740)
    arte = Image.new('RGB',tamanho_arte,cor_fundo)
    W,H = arte.size
    w,h= im.size
    im = im.resize((int(w/1.5),int(h/1.5)))
    im = im.copy()
    arte.paste(im,(700,600))

    font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
    msg = f'Sumário da partida'
    draw = ImageDraw.Draw(arte)
    w, h = draw.textsize(msg,spacing=20,font=font)
    draw.text((900,100),msg, fill='white',spacing= 20,font=font)

    font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
    msg = f'{home_team} - {away_team}'
    draw = ImageDraw.Draw(arte)
    w, h = draw.textsize(msg,spacing=20,font=font)
    draw.text((900,400),msg, fill='white',spacing= 20,font=font)

    im = Image.open('Arquivos/legenda-linha.png')
    w,h = im.size
    im = im.resize((int(w/5),int(h/5)))
    im = im.copy()
    arte.paste(im,(1200,600))

    font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
    msg = f'Mandante'
    draw = ImageDraw.Draw(arte)
    draw.text((1400,640),msg, fill='white',spacing= 30,font=font)


    font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
    msg = f'Visitante'
    draw = ImageDraw.Draw(arte)
    draw.text((1770,640),msg, fill='white',spacing= 30,font=font)


    fot =Image.open('Logos/Copy of pro_branco.png')
    w,h = fot.size
    fot = fot.resize((int(w/2),int(h/2)))
    fot = fot.copy()
    arte.paste(fot,(1300,2300),fot)
    arte.save(f'content/quadro_summary_{home_team}_{away_team}.png',quality=95,facecolor='#2C2B2B')
    st.image(f'content/quadro_summary_{home_team}_{away_team}.png')
    st.markdown(get_binary_file_downloader_html(f'content/quadro_summary_{home_team}_{away_team}.png', 'Imagem'), unsafe_allow_html=True)
