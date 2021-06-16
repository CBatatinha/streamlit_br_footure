import streamlit as st
import pandas as pd 
import requests as requests
import io
import os
import json
from pandas.io.json import json_normalize
import glob as glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, ConnectionPatch,Ellipse
import PIL
from PIL import Image
from PIL import Image, ImageDraw, ImageFilter
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageOps
from random import randint
from time import sleep
import matplotlib.font_manager as font_manager
import matplotlib as mpl 
import pylab as pl
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
#------------------------------------------------------------------------------------------------------- 
st.title('Footure Brasileirão v1.2')
menu=['Home','Gráficos jogadores (Partida)','Gráficos jogadores (Total)','Gráficos times (Partida)']
choice=st.sidebar.selectbox('Menu',menu)
if choice == 'Home':
   st.markdown('Ferramenta criada pelo departamento de análise de dados do Footure PRO para visualizações  \n'
                'Navegue pelas abas no menu para obter os gráficos de interesse.  \n'
                'Temporada 2021 tem até a 2° rodada.  \n'
                'Euro já disponível.  \n'
                '**Obs: Passes mais valiosos ainda não ta disponivel ** ')

org_2020= "https://drive.google.com/file/d/1-14BD_oWbQhuT3fNiC5P3cwJjqsAkX7S/view?usp=sharing"
file_id_1= org_2020.split('/')[-2]
url_2020='https://drive.google.com/uc?export=download&id=' + file_id_1
gdown.download(url_2020,'br2020.csv',quiet=True)
org_2021= "https://drive.google.com/file/d/1o_FqfT_hzU3gFzr7WFHpZZ9Sv5a1ZgDo/view?usp=sharing"
file_id_2= org_2021.split('/')[-2]
url_2021='https://drive.google.com/uc?export=download&id=' + file_id_2
gdown.download(url_2021,'br2021.csv',quiet=True)
org_america= "https://drive.google.com/file/d/1o_FqfT_hzU3gFzr7WFHpZZ9Sv5a1ZgDo/view?usp=sharing"
file_id_america= org_america.split('/')[-2]
url_america='https://drive.google.com/uc?export=download&id=' + file_id_america
gdown.download(url_america,'america2021.csv',quiet=True)

org_euro= "https://drive.google.com/file/d/1hIepn0XBk6prx-bxdQN0TkqKvwDlvbrr/view?usp=sharing"
file_id_euro= org_euro.split('/')[-2]
url_euro='https://drive.google.com/uc?export=download&id=' + file_id_euro
gdown.download(url_euro,'euro2021.csv',quiet=True)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

if choice == 'Gráficos jogadores (Partida)':
   st.subheader('Plote os gráficos individuais dos jogadores em uma partida do campeonato')
   lista_temporada=['2020','2021','Euro 2021','Copa America 2021']
   temporada=st.selectbox('Selecione a temporada',lista_temporada)
   if temporada == '2020':   
      df = pd.read_csv('br2020.csv',encoding = "utf-8-sig")
   if temporada == '2021':
      df = pd.read_csv('br2021.csv',encoding = "utf-8-sig")
   if temporada == '2021':
      df = pd.read_csv('br2021.csv',encoding = "utf-8-sig")
   if temporada == 'Euro 2021':
      df = pd.read_csv('euro2021.csv',encoding = "utf-8-sig")
      teams_dict= {'Brazil':'Brasil','Paraguay':'Paraguai','Uruguay':'Uruguai','Colombia':'Colômbia','Ecuador':'Equador',
                   'Italy':'Itália','Switzerland':'Suíça','Turkey':'Turquia','Wales':'Gales','Belgium':'Bélgica','Denmark':'Dinamarca',
                   'Finland':'Finlândia','Russia':'Rússia','Netherlands':'Holanda','North Macedonia':'Macedônia do norte',
                   'Ukraine':'Ucrânia','Poland':'Polônia','Slovakia':'Eslováquia','Spain':'Espanha','Sweden':'Suécia',
                   'Croatia':'Croácia','Czech Republic':'Rep. Tcheca','England':'Inglaterra','Scotland':'Escócia',
                   'France':'França','Germany':'Alemanha','Hungary':'Hungria','Austria':'Áustria','Portugal':'Portugal'}
      df['hometeam']=df['hometeam'].map(teams_dict)
      df['awayteam']=df['awayteam'].map(teams_dict)
   if temporada == 'Copa America 2021':
      df = pd.read_csv('america2021.csv',encoding = "utf-8-sig")
      teams_dict= {'Brazil':'Brasil','Paraguay':'Paraguai','Uruguay':'Uruguai','Colombia':'Colômbia','Ecuador':'Equador',
                   'Italy':'Itália','Switzerland':'Suíça','Turkey':'Turquia','Wales':'Gales','Belgium':'Bélgica','Denmark':'Dinamarca',
                   'Finland':'Finlândia','Russia':'Rússia','Netherlands':'Holanda','North Macedonia':'Macedônia do norte',
                   'Ukraine':'Ucrânia','Poland':'Polônia','Slovakia':'Eslováquia','Spain':'Espanha','Sweden':'Suécia',
                   'Croatia':'Croácia','Czech Republic':'Rep. Tcheca','England':'Inglaterra','Scotland':'Escócia',
                   'France':'França','Germany':'Alemanha','Hungary':'Hungria'}
      df['hometeam']=df['hometeam'].map(teams_dict)
      df['awayteam']=df['awayteam'].map(teams_dict)
   nav1,nav2 = st.beta_columns(2)
   with nav1:
        home_team=st.selectbox('Time da casa',sorted(list(df['hometeam'].unique())))
   with nav2:
        away_team=st.selectbox('Time de fora',sorted(list(df['awayteam'].unique())))
   match=df[(df['hometeam']==home_team)&(df['awayteam']==away_team)].reset_index(drop=True)
   jogador=st.selectbox('Escolha o jogador',list(match['name'].unique()))
   df_jogador=match[(match['name']==jogador)].reset_index(drop=True)
   lista_graficos=['Heatmap','Recepções','Passes','Ações Defensivas','Passes mais frequentes']
   grafico=st.selectbox('Escolha o gráfico',lista_graficos)
   if grafico == 'Heatmap':
      def heatmap(df):
        heatmap=df[df['isTouch']==True].reset_index(drop=True)
        cor_fundo = '#000000'
        fig, ax = plt.subplots(figsize=(20,10))
        pitch = Pitch(pitch_type='uefa', figsize=(20,10),pitch_color=cor_fundo,
                        stripe=False, line_zorder=2)
        pitch.draw(ax=ax)
        cor_ponto = 'black' 
        sns.kdeplot(heatmap["x"],heatmap["y"], shade=True, n_levels=250,cmap='CMRmap')
          #Spectral_r
        ax.set_ylim(0,68)
        ax.set_xlim(0,105)
        plt.savefig(f'calor_{jogador}.jpg',quality=95,facecolor=cor_fundo)
        im = Image.open(f'calor_{jogador}.jpg')
        
        # cor_fundo = '#2c2b2b'
        tamanho_arte = (3000, 2740)
        arte = Image.new('RGB',tamanho_arte,'#2C2B2B')
        W,H = arte.size
        im = im.rotate(90,expand=6)
        border = (115, 375, 110, 430) # left, up, right, bottom
        im = ImageOps.crop(im, border)
        w,h= im.size
        im = im.resize((int(w*1.2),int(h*1.2)))
        im = im.copy()
        w,h= im.size
        arte.paste(im,(330,680))

        font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
        msg = f'Mapa de calor'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,100),msg, fill='white',spacing= 20,font=font)

        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{home_team}- {away_team}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,300),msg, fill='white',spacing= 20,font=font)

        # acerto=len(passe_certo)
        # total=(len(passe_certo)+len(passe_errado))
        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{jogador}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,500),msg, fill='white',spacing= 20,font=font)

        fot =Image.open('Logos/Copy of pro_branco.png')
        w,h = fot.size
        fot = fot.resize((int(w/1.5),int(h/1.5)))
        fot = fot.copy()
        arte.paste(fot,(1770,1680),fot)

        if df['hometeamid'][0]==df['teamId'][0]:
          team=(df['hometeam'][0])
        else:
          team=(df['awayteam'][0])

        times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
        logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
        try:
          r = requests.get(logo_url)
          im_bt = r.content
          image_file = io.BytesIO(im_bt)
          im = Image.open(image_file)
          w,h = im.size
          im = im.resize((int(w*2.5),int(h*2.5)))
          im = im.copy()
          arte.paste(im,(2500,100),im)
        except:
          r = requests.get(logo_url)
          im_bt = r.content
          image_file = io.BytesIO(im_bt)
          im = Image.open(image_file)
          w,h = im.size
          im = im.resize((int(w*2.5),int(h*2.5)))
          im = im.copy()
          arte.paste(im,(2500,100))

        arte.save(f'content/quadro_calor_{jogador}.png',quality=95,facecolor='#2C2B2B')
        st.image(f'content/quadro_calor_{jogador}.png')
        st.markdown(get_binary_file_downloader_html(f'content/quadro_calor_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
      heatmap(df_jogador)
   if grafico == 'Recepções':
      def recepcao(df):
          rec=df[((df['type_displayName']=='Pass')&(df['outcomeType_displayName']=='Successful')) & (df['receiver']==jogador)].reset_index(drop=True)
          y = list(rec['endY'])
          x = list(rec['endX'])
          cor_fundo = '#2c2b2b'
          fig, ax = plt.subplots(figsize=(15,10))
          pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                          stripe=False, line_zorder=2)
          pitch.draw(ax=ax)
          from matplotlib.colors import LinearSegmentedColormap

          cmap = LinearSegmentedColormap.from_list('name', [cor_fundo, '#F43B87'])
          plt.hist2d(x,y, bins=[np.arange(0, 120, 10), np.arange(0, 120, 10)], cmap=cmap)
          plt.savefig(f'content/recepção_{jogador}.png',dpi=300,facecolor=cor_fundo)
          im = Image.open(f'content/recepção_{jogador}.png')
          # cor_fundo = '#2c2b2b'
          tamanho_arte = (3000, 2740)
          arte = Image.new('RGB',tamanho_arte,cor_fundo)
          W,H = arte.size
          im = im.rotate(90,expand=1)
          border = (1235, 1130, 00, 0) # left, up, right, bottom
          im = ImageOps.crop(im, border)
          w,h= im.size
          im = im.resize((int(w/1.2),int(h/1.2)))
          im = im.copy()
          w,h= im.size
          arte.paste(im,(330,600))

          font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
          msg = f'Recepções'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,100),msg, fill='white',spacing= 20,font=font)

          font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
          msg = f'{home_team}- {away_team}'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,300),msg, fill='white',spacing= 20,font=font)

          # acerto=len(passe_certo)
          # total=(len(passe_certo)+len(passe_errado))
          font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
          msg = f'{jogador}'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,500),msg, fill='white',spacing= 20,font=font)

          fot =Image.open('Logos/Copy of pro_branco.png')
          w,h = fot.size
          fot = fot.resize((int(w/1.5),int(h/1.5)))
          fot = fot.copy()
          arte.paste(fot,(1870,1980),fot)

          if df_jogador['hometeamid'][0]==df_jogador['teamId'][0]:
            team=(df_jogador['hometeam'][0])
          else:
            team=(df_jogador['awayteam'][0])

          times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
          logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
          try:
            r = requests.get(logo_url)
            im_bt = r.content
            image_file = io.BytesIO(im_bt)
            im = Image.open(image_file)
            w,h = im.size
            im = im.resize((int(w*2.5),int(h*2.5)))
            im = im.copy()
            arte.paste(im,(2500,100),im)
          except:
            r = requests.get(logo_url)
            im_bt = r.content
            image_file = io.BytesIO(im_bt)
            im = Image.open(image_file)
            w,h = im.size
            im = im.resize((int(w*2.5),int(h*2.5)))
            im = im.copy()
            arte.paste(im,(2500,100))

          arte.save(f'content/quadro_recep_{jogador}.png',quality=95,facecolor='#2C2B2B')
          st.image(f'content/quadro_recep_{jogador}.png')
          st.markdown(get_binary_file_downloader_html(f'content/quadro_recep_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
      recepcao(match)
   if grafico == 'Passes':
      tipos_passe=['Passe Simples','Infiltrado','Chave','Cruzamento','Assistência','Escanteio','Falta','Progressivo']
      lista_passes=st.selectbox('Escolha os passes',tipos_passe)
      def passes(df1,df2):
          cor_fundo = '#2c2b2b'
          fig, ax = plt.subplots(figsize=(15,10))
          pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                          stripe=False, line_zorder=2)
          pitch.draw(ax=ax)
          certo=df1
          errado=df2
          def plot_scatter_df(df,cor,zo):
              pitch.scatter(df.endX, df.endY, s=200, edgecolors=cor,lw=2, c=cor_fundo, zorder=zo+1, ax=ax)
              # plt.scatter(data=df, x='to_x',y='to_y',color=cor,zorder=zo+1,label='df',edgecolors='white',s=200):
              x_inicial = df['x']
              y_inicial = df['y']
              x_final = df['endX']
              y_final = df['endY']
              lc1 = pitch.lines(x_inicial, y_inicial,
                            x_final,y_final,
                            lw=5, transparent=True, comet=True,color=cor, ax=ax,zorder=zo)
          plot_scatter_df(certo,'#00FF79',12)
          plot_scatter_df(errado,'#FD2B2C',9)
          plt.show()
          plt.savefig(f'content/passe_{jogador}.png',dpi=300,facecolor=cor_fundo)
          im = Image.open(f'content/passe_{jogador}.png')
          cor_fundo = '#2c2b2b'
          tamanho_arte = (3000, 2740)
          arte = Image.new('RGB',tamanho_arte,cor_fundo)
          W,H = arte.size
          im = im.rotate(90,expand=5)
          w,h= im.size
          im = im.resize((int(w/2),int(h/2)))
          im = im.copy()
          w,h= im.size
          arte.paste(im,(100,400))

          font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
          msg = f'{lista_passes}'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,100),msg, fill='white',spacing= 20,font=font)

          font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
          msg = f'{home_team}- {away_team}'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,300),msg, fill='white',spacing= 20,font=font)

          acerto=len(passe_certo)
          total=(len(passe_certo)+len(passe_errado))
          font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
          msg = f'{jogador}: {acerto}/{total}'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,500),msg, fill='white',spacing= 20,font=font)

          im = Image.open('Arquivos/legenda-acerto-erro.png')
          w,h = im.size
          im = im.resize((int(w/5),int(h/5)))
          im = im.copy()
          arte.paste(im,(330,2350))

          font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
          msg = f'Certo'
          draw = ImageDraw.Draw(arte)
          draw.text((600,2400),msg, fill='white',spacing= 30,font=font)


          font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
          msg = f'Errado'
          draw = ImageDraw.Draw(arte)
          draw.text((920,2400),msg, fill='white',spacing= 30,font=font)

          fot =Image.open('Logos/Copy of pro_branco.png')
          w,h = fot.size
          fot = fot.resize((int(w/1.5),int(h/1.5)))
          fot = fot.copy()
          arte.paste(fot,(1870,1880),fot)

          if df_jogador['hometeamid'][0]==df_jogador['teamId'][0]:
            team=(df_jogador['hometeam'][0])
          else:
            team=(df_jogador['awayteam'][0])

          times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
          logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
          try:
            r = requests.get(logo_url)
            im_bt = r.content
            image_file = io.BytesIO(im_bt)
            im = Image.open(image_file)
            w,h = im.size
            im = im.resize((int(w*2.5),int(h*2.5)))
            im = im.copy()
            arte.paste(im,(2500,100),im)
          except:
            r = requests.get(logo_url)
            im_bt = r.content
            image_file = io.BytesIO(im_bt)
            im = Image.open(image_file)
            w,h = im.size
            im = im.resize((int(w*2.5),int(h*2.5)))
            im = im.copy()
            arte.paste(im,(2500,100))


          arte.save(f'content/quadro_{lista_passes}_{jogador}.png',quality=95,facecolor='#2C2B2B')
          st.image(f'content/quadro_{lista_passes}_{jogador}.png')
          st.markdown(get_binary_file_downloader_html(f'content/quadro_{lista_passes}_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
      if 'Passe Simples' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['events']=='Pass')&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['events']=='Pass')&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Infiltrado' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['TB']==1)&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['TB']==1)&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Chave' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['KP']==1)&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['KP']==1)&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)  
      if 'Cruzamento' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['events']=='cross')&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['events']=='cross')&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Assistência' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['Assist']==1)&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['Assist']==1)&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Falta' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&((df_jogador['events']=='freekick_short')|(df_jogador['events']=='freekick_crossed'))&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&((df_jogador['events']=='freekick_short')|(df_jogador['events']=='freekick_crossed'))&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Escanteio' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&((df_jogador['events']=='corner_short')|(df_jogador['events']=='corner_crossed'))&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&((df_jogador['events']=='corner_short')|(df_jogador['events']=='corner_crossed'))&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Progressivo' in lista_passes:
          df_jogador=df_jogador[(df_jogador['type_displayName']=='Pass')].reset_index(drop=True)
          df_jogador['dist1'] = np.sqrt((105-df_jogador.x)**2 + (34-df_jogador.y)**2)
          df_jogador['dist2'] = np.sqrt((105-df_jogador.endX)**2 + (34-df_jogador.endY)**2)
          df_jogador['distdiff'] = df_jogador['dist1'] - df_jogador['dist2']
          pass1 = df_jogador.query("(x<52.5)&(endX<52.5)&(distdiff>=30)")
          pass2 = df_jogador.query("(x<52.5)&(endX>52.5)&(distdiff>=15)")
          pass3 = df_jogador.query("(x>52.5)&(endX>52.5)&(distdiff>=10)")
          pass1 = pass1.append(pass2)
          pass1 = pass1.append(pass3)
          passe_certo=pass1[(pass1['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=pass1[(pass1['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
   if grafico == 'Ações Defensivas':
      tipos_defesa=['Desarme','Interceptação','Corte','Bloqueio','Aéreo','Duelo']
      lista_defesa=st.multiselect('Escolha as ações',tipos_defesa)
      dct_defense={'Desarme':'Tackle','Interceptação':'Interception','Corte':'Clearance',
                   'Bloqueio':'BlockedPass','Aéreo':'Aerial','Duelo':'Challenge'}
      defesa=[dct_defense[k] for k in lista_defesa]
      defesa_certo=df_jogador[(df_jogador['type_displayName'].isin(defesa))&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
      defesa_errado=df_jogador[(df_jogador['type_displayName'].isin(defesa))&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
      cor_fundo = '#2c2b2b'
      fig, ax = plt.subplots(figsize=(15,10))
      pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                      stripe=False, line_zorder=2)
      pitch.draw(ax=ax)
      zo=12
      plt.scatter(data=defesa_certo, x='x',y='y',color='#00FF79',zorder=zo+1)
      plt.scatter(data=defesa_errado, x='x',y='y',color='#FD2B2C',zorder=zo+1)
      defensivo=df_jogador[(df_jogador['type_displayName'].isin(defesa))].reset_index(drop=True)
      plt.axvline(x=defensivo['x'].mean(),ymin=0.05, ymax=0.95, color='#7AB5B7', linestyle='--',lw=2)
      plt.savefig(f'content/defesa_{jogador}.png',dpi=300,facecolor=cor_fundo)
      im = Image.open(f'content/defesa_{jogador}.png')
      cor_fundo = '#2c2b2b'
      tamanho_arte = (3000, 2740)
      arte = Image.new('RGB',tamanho_arte,cor_fundo)
      W,H = arte.size
      im = im.rotate(90,expand=5)
      w,h= im.size
      im = im.resize((int(w/2),int(h/2)))
      im = im.copy()
      w,h= im.size
      arte.paste(im,(100,400))

      font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
      msg = f'{grafico}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,100),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{home_team}- {away_team}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,300),msg, fill='white',spacing= 20,font=font)

      acerto=len(defesa_certo)
      total=(len(defesa_certo)+len(defesa_errado))
      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{jogador}: {acerto}/{total}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,500),msg, fill='white',spacing= 20,font=font)

      im = Image.open('Arquivos/legenda-acerto-erro.png')
      w,h = im.size
      im = im.resize((int(w/5),int(h/5)))
      im = im.copy()
      arte.paste(im,(330,2350))

      font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
      msg = f'Certo'
      draw = ImageDraw.Draw(arte)
      draw.text((600,2400),msg, fill='white',spacing= 30,font=font)


      font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
      msg = f'Errado'
      draw = ImageDraw.Draw(arte)
      draw.text((920,2400),msg, fill='white',spacing= 30,font=font)

      fot =Image.open('Logos/Copy of pro_branco.png')
      w,h = fot.size
      fot = fot.resize((int(w/1.5),int(h/1.5)))
      fot = fot.copy()
      arte.paste(fot,(1870,1880),fot)

      if df_jogador['hometeamid'][0]==df_jogador['teamId'][0]:
        team=(df_jogador['hometeam'][0])
      else:
        team=(df_jogador['awayteam'][0])

      times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
      logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
      try:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100),im)
      except:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100))


      arte.save(f'content/quadro_{grafico}_{jogador}.png',quality=95,facecolor='#2C2B2B')
      st.image(f'content/quadro_{grafico}_{jogador}.png')
      st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
   if grafico == 'Passes mais frequentes':
      df_passe_plot= df_jogador[(df_jogador.events.isin(['Pass','cross']))&
         (df_jogador.outcomeType_displayName=='Successful')].reset_index(drop=True)
      df_passe_plot=df_passe_plot[['x','y','endX','endY']]
      valores = df_passe_plot.values
      passes_total = len(df_passe_plot)
      max_cluster = int(len(df_passe_plot)/4)
      min_cluster = int(len(df_passe_plot)/10)
      dic_geral_clusters = []
      df_passe = df_passe_plot

      dic_sill = {}
      for i in range(min_cluster, max_cluster):
        km = KMeans(n_clusters=i)
        km.fit(valores)
        label = km.predict(valores)
        sill = silhouette_score(valores,label)
        dic_sill.update({i:sill})

      df_sill = pd.DataFrame(dic_sill,index=[0]).transpose().reset_index()
      n_cluster = df_sill.sort_values(0,ascending=False).reset_index()['index'][0]
      valor =  df_sill.sort_values(0,ascending=False).reset_index()[0][0]
      print(f'{n_cluster}:{valor}')
      dic_geral_clusters.append({'metodo':'kmeans','n_cluster':'n_cluster','acerto':valor})

      km = KMeans(
          n_clusters=n_cluster, init='random',
          n_init=1, max_iter=300, 
          tol=1e-04, random_state=0
      )
      y_km = km.fit_predict(valores)

      # cor_fundo='#2C2B2B'
      df_passe['cluster'] = y_km
      df_passe['quantidade'] = 0
      cluster = df_passe.groupby('cluster')['quantidade'].count().reset_index().sort_values('quantidade',ascending=False).reset_index(drop=True)
      lista_cluster = list(cluster['cluster'])[0:3]
      df_plot = df_passe[df_passe['cluster'].isin(lista_cluster)].reset_index(drop=True)
      # df_plot
      cor_fundo = '#2c2b2b'
      fig, ax = plt.subplots(figsize=(15,10))
      pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                      stripe=False, line_zorder=1)
      pitch.draw(ax=ax)
      def plot_scatter_df(df,cor,zo):
        pitch.scatter(df.endX, df.endY, s=200, edgecolors=cor,lw=2, c=cor_fundo, zorder=zo+1, ax=ax)
        # plt.scatter(data=df, x='to_x',y='to_y',color=cor,zorder=zo+1,label='df',edgecolors='white',s=200)
        for linha in range(len(df)):
          x_inicial = df['x'][linha]
          y_inicial = df['y'][linha]
          x_final = df['endX'][linha]
          y_final = df['endY'][linha]
          lc1 = pitch.lines(x_inicial, y_inicial,
                        x_final, y_final,
                        lw=5, transparent=True, comet=True,
                        color=cor, ax=ax,zorder=zo)
      
      lista_cor = ['#FF4E63','#8D9713','#00A6FF']
      for clus,cor in zip(lista_cluster,lista_cor):
        df = (df_plot[df_plot['cluster'] == clus].reset_index())
        df['cor'] = cor 
        plot_scatter_df(df,cor,2)
      plt.show()
      plt.savefig(f'content/cluster_{jogador}.png',dpi=300,facecolor=cor_fundo)
      im = Image.open(f'content/cluster_{jogador}.png')
      cor_fundo = '#2c2b2b'
      tamanho_arte = (3000, 2740)
      arte = Image.new('RGB',tamanho_arte,cor_fundo)
      W,H = arte.size
      im = im.rotate(90,expand=5)
      w,h= im.size
      im = im.resize((int(w/2),int(h/2)))
      im = im.copy()
      w,h= im.size
      arte.paste(im,(100,400))

      font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
      msg = f'Cluster de passes'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,100),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{home_team}- {away_team}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,300),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{jogador}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,500),msg, fill='white',spacing= 20,font=font)

      fot =Image.open('Logos/Copy of pro_branco.png')
      w,h = fot.size
      fot = fot.resize((int(w/1.5),int(h/1.5)))
      fot = fot.copy()
      arte.paste(fot,(1870,1880),fot)

      if df_jogador['hometeamid'][0]==df_jogador['teamId'][0]:
        team=(df_jogador['hometeam'][0])
      else:
        team=(df_jogador['awayteam'][0])

      times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
      logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
      try:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100))
      except:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100))


      arte.save(f'content/quadro_{grafico}_{jogador}.png',quality=95,facecolor='#2C2B2B')
      st.image(f'content/quadro_{grafico}_{jogador}.png')
      st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
if choice == 'Gráficos jogadores (Total)':
   st.subheader('Plote os gráficos individuais dos jogadores em todas as partidas')
   lista_temporada=['2020','2021','Euro 2021','Copa America 2021']
   temporada=st.selectbox('Selecione a temporada',lista_temporada)
   if temporada == '2020':   
      df = pd.read_csv('br2020.csv',encoding = "utf-8-sig")
   if temporada == '2021':
      df = pd.read_csv('br2021.csv',encoding = "utf-8-sig")
   if temporada == '2021':
      df = pd.read_csv('br2021.csv',encoding = "utf-8-sig")
   if temporada == 'Euro 2021':
      df = pd.read_csv('euro2021.csv',encoding = "utf-8-sig")
      teams_dict= {'Brazil':'Brasil','Paraguay':'Paraguai','Uruguay':'Uruguai','Colombia':'Colômbia','Ecuador':'Equador',
                   'Italy':'Itália','Switzerland':'Suíça','Turkey':'Turquia','Wales':'Gales','Belgium':'Bélgica','Denmark':'Dinamarca',
                   'Finland':'Finlândia','Russia':'Rússia','Netherlands':'Holanda','North Macedonia':'Macedônia do norte',
                   'Ukraine':'Ucrânia','Poland':'Polônia','Slovakia':'Eslováquia','Spain':'Espanha','Sweden':'Suécia',
                   'Croatia':'Croácia','Czech Republic':'Rep. Tcheca','England':'Inglaterra','Scotland':'Escócia',
                   'France':'França','Germany':'Alemanha','Hungary':'Hungria','Austria':'Áustria','Portugal':'Portugal'}
      df['hometeam']=df['hometeam'].map(teams_dict)
      df['awayteam']=df['awayteam'].map(teams_dict)
   if temporada == 'Copa America 2021':
      df = pd.read_csv('america2021.csv',encoding = "utf-8-sig")
      teams_dict= {'Brazil':'Brasil','Paraguay':'Paraguai','Uruguay':'Uruguai','Colombia':'Colômbia','Ecuador':'Equador',
                   'Italy':'Itália','Switzerland':'Suíça','Turkey':'Turquia','Wales':'Gales','Belgium':'Bélgica','Denmark':'Dinamarca',
                   'Finland':'Finlândia','Russia':'Rússia','Netherlands':'Holanda','North Macedonia':'Macedônia do norte',
                   'Ukraine':'Ucrânia','Poland':'Polônia','Slovakia':'Eslováquia','Spain':'Espanha','Sweden':'Suécia',
                   'Croatia':'Croácia','Czech Republic':'Rep. Tcheca','England':'Inglaterra','Scotland':'Escócia',
                   'France':'França','Germany':'Alemanha','Hungary':'Hungria'}
      df['hometeam']=df['hometeam'].map(teams_dict)
      df['awayteam']=df['awayteam'].map(teams_dict)
   team=st.selectbox('Escolha o time',sorted(list(df['hometeam'].unique())))
   match=df[((df['hometeam']==team) & (df['hometeamid']==df.teamId)) | ((df['awayteam']==team) & (df['awayteamid']==df.teamId))].reset_index(drop=True)
   jogador=st.selectbox('Escolha o jogador',list(match['name'].unique()))
   df_jogador=match[(match['name']==jogador)].reset_index(drop=True)
   lista_graficos=['Heatmap','Recepções','Passes','Ações Defensivas','Passes mais frequentes','Sonar Inverso de chutes']
   grafico=st.selectbox('Escolha o gráfico',lista_graficos)
   if grafico == 'Heatmap':
      def heatmap(df):
        heatmap=df[df['isTouch']==True].reset_index(drop=True)
        cor_fundo = '#000000'
        fig, ax = plt.subplots(figsize=(20,10))
        pitch = Pitch(pitch_type='uefa', figsize=(20,10),pitch_color=cor_fundo,
                        stripe=False, line_zorder=2)
        pitch.draw(ax=ax)
        cor_ponto = 'black' 
        sns.kdeplot(heatmap["x"],heatmap["y"], shade=True, n_levels=250,cmap='CMRmap')
          #Spectral_r
        ax.set_ylim(0,68)
        ax.set_xlim(0,105)
        plt.savefig(f'calor_{jogador}.jpg',quality=95,facecolor=cor_fundo)
        im = Image.open(f'calor_{jogador}.jpg')
        
        # cor_fundo = '#2c2b2b'
        tamanho_arte = (3000, 2740)
        arte = Image.new('RGB',tamanho_arte,'#2C2B2B')
        W,H = arte.size
        im = im.rotate(90,expand=6)
        border = (115, 375, 110, 430) # left, up, right, bottom
        im = ImageOps.crop(im, border)
        w,h= im.size
        im = im.resize((int(w*1.2),int(h*1.2)))
        im = im.copy()
        w,h= im.size
        arte.paste(im,(330,680))

        font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
        msg = f'Mapa de calor'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,100),msg, fill='white',spacing= 20,font=font)

        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{team}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,300),msg, fill='white',spacing= 20,font=font)

        # acerto=len(passe_certo)
        # total=(len(passe_certo)+len(passe_errado))
        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{jogador}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,500),msg, fill='white',spacing= 20,font=font)

        fot =Image.open('Logos/Copy of pro_branco.png')
        w,h = fot.size
        fot = fot.resize((int(w/1.5),int(h/1.5)))
        fot = fot.copy()
        arte.paste(fot,(1770,1680),fot)

        times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
        logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
        try:
          r = requests.get(logo_url)
          im_bt = r.content
          image_file = io.BytesIO(im_bt)
          im = Image.open(image_file)
          w,h = im.size
          im = im.resize((int(w*2.5),int(h*2.5)))
          im = im.copy()
          arte.paste(im,(2500,100),im)
        except:
          r = requests.get(logo_url)
          im_bt = r.content
          image_file = io.BytesIO(im_bt)
          im = Image.open(image_file)
          w,h = im.size
          im = im.resize((int(w*2.5),int(h*2.5)))
          im = im.copy()
          arte.paste(im,(2500,100))

        arte.save(f'content/quadro_calor_{jogador}.png',quality=95,facecolor='#2C2B2B')
        st.image(f'content/quadro_calor_{jogador}.png')
        st.markdown(get_binary_file_downloader_html(f'content/quadro_calor_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
      heatmap(df_jogador)
   if grafico == 'Recepções':
      def recepcao(df):
          rec=df[((df['type_displayName']=='Pass')&(df['outcomeType_displayName']=='Successful')) & (df['receiver']==jogador)].reset_index(drop=True)
          y = list(rec['endY'])
          x = list(rec['endX'])
          cor_fundo = '#2c2b2b'
          fig, ax = plt.subplots(figsize=(15,10))
          pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                          stripe=False, line_zorder=2)
          pitch.draw(ax=ax)
          from matplotlib.colors import LinearSegmentedColormap

          cmap = LinearSegmentedColormap.from_list('name', [cor_fundo, '#F43B87'])
          plt.hist2d(x,y, bins=[np.arange(0, 120, 10), np.arange(0, 120, 10)], cmap=cmap)
          plt.savefig(f'content/recepção_{jogador}.png',dpi=300,facecolor=cor_fundo)
          im = Image.open(f'content/recepção_{jogador}.png')
          # cor_fundo = '#2c2b2b'
          tamanho_arte = (3000, 2740)
          arte = Image.new('RGB',tamanho_arte,cor_fundo)
          W,H = arte.size
          im = im.rotate(90,expand=1)
          border = (1235, 1130, 00, 0) # left, up, right, bottom
          im = ImageOps.crop(im, border)
          w,h= im.size
          im = im.resize((int(w/1.2),int(h/1.2)))
          im = im.copy()
          w,h= im.size
          arte.paste(im,(330,600))

          font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
          msg = f'Recepções'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,100),msg, fill='white',spacing= 20,font=font)

          font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
          msg = f'{team}'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,300),msg, fill='white',spacing= 20,font=font)

          # acerto=len(passe_certo)
          # total=(len(passe_certo)+len(passe_errado))
          font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
          msg = f'{jogador}'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,500),msg, fill='white',spacing= 20,font=font)

          fot =Image.open('Logos/Copy of pro_branco.png')
          w,h = fot.size
          fot = fot.resize((int(w/1.5),int(h/1.5)))
          fot = fot.copy()
          arte.paste(fot,(1870,1980),fot)

          times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
          logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
          try:
            r = requests.get(logo_url)
            im_bt = r.content
            image_file = io.BytesIO(im_bt)
            im = Image.open(image_file)
            w,h = im.size
            im = im.resize((int(w*2.5),int(h*2.5)))
            im = im.copy()
            arte.paste(im,(2500,100),im)
          except:
            r = requests.get(logo_url)
            im_bt = r.content
            image_file = io.BytesIO(im_bt)
            im = Image.open(image_file)
            w,h = im.size
            im = im.resize((int(w*2.5),int(h*2.5)))
            im = im.copy()
            arte.paste(im,(2500,100))

          arte.save(f'content/quadro_recep_{jogador}.png',quality=95,facecolor='#2C2B2B')
          st.image(f'content/quadro_recep_{jogador}.png')
          st.markdown(get_binary_file_downloader_html(f'content/quadro_recep_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
      recepcao(match)
   if grafico == 'Passes':
      tipos_passe=['Simples','Infiltrado','Chave','Cruzamento','Assistência','Escanteio','Falta','Progressivo']
      lista_passes=st.selectbox('Escolha os passes',tipos_passe)
      def passes(df1,df2):
          cor_fundo = '#2c2b2b'
          fig, ax = plt.subplots(figsize=(15,10))
          pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                          stripe=False, line_zorder=2)
          pitch.draw(ax=ax)
          certo=df1
          errado=df2
          def plot_scatter_df(df,cor,zo):
              pitch.scatter(df.endX, df.endY, s=200, edgecolors=cor,lw=2, c=cor_fundo, zorder=zo+1, ax=ax)
              # plt.scatter(data=df, x='to_x',y='to_y',color=cor,zorder=zo+1,label='df',edgecolors='white',s=200):
              x_inicial = df['x']
              y_inicial = df['y']
              x_final = df['endX']
              y_final = df['endY']
              lc1 = pitch.lines(x_inicial, y_inicial,
                            x_final,y_final,
                            lw=5, transparent=True, comet=True,color=cor, ax=ax,zorder=zo)
          plot_scatter_df(certo,'#00FF79',12)
          plot_scatter_df(errado,'#FD2B2C',9)
          plt.show()
          plt.savefig(f'content/passe_{jogador}.png',dpi=300,facecolor=cor_fundo)
          im = Image.open(f'content/passe_{jogador}.png')
          cor_fundo = '#2c2b2b'
          tamanho_arte = (3000, 2740)
          arte = Image.new('RGB',tamanho_arte,cor_fundo)
          W,H = arte.size
          im = im.rotate(90,expand=5)
          w,h= im.size
          im = im.resize((int(w/2),int(h/2)))
          im = im.copy()
          w,h= im.size
          arte.paste(im,(100,400))

          font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
          msg = f'{lista_passes}'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,100),msg, fill='white',spacing= 20,font=font)

          font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
          msg = f'{team}'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,300),msg, fill='white',spacing= 20,font=font)

          acerto=len(passe_certo)
          total=(len(passe_certo)+len(passe_errado))
          font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
          msg = f'{jogador}: {acerto}/{total}'
          draw = ImageDraw.Draw(arte)
          w, h = draw.textsize(msg,spacing=20,font=font)
          draw.text((330,500),msg, fill='white',spacing= 20,font=font)

          im = Image.open('Arquivos/legenda-acerto-erro.png')
          w,h = im.size
          im = im.resize((int(w/5),int(h/5)))
          im = im.copy()
          arte.paste(im,(330,2350))

          font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
          msg = f'Certo'
          draw = ImageDraw.Draw(arte)
          draw.text((600,2400),msg, fill='white',spacing= 30,font=font)


          font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
          msg = f'Errado'
          draw = ImageDraw.Draw(arte)
          draw.text((920,2400),msg, fill='white',spacing= 30,font=font)

          fot =Image.open('Logos/Copy of pro_branco.png')
          w,h = fot.size
          fot = fot.resize((int(w/1.5),int(h/1.5)))
          fot = fot.copy()
          arte.paste(fot,(1870,1880),fot)

          times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
          logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
          try:
            r = requests.get(logo_url)
            im_bt = r.content
            image_file = io.BytesIO(im_bt)
            im = Image.open(image_file)
            w,h = im.size
            im = im.resize((int(w*2.5),int(h*2.5)))
            im = im.copy()
            arte.paste(im,(2500,100),im)
          except:
            pass

          arte.save(f'content/quadro_{lista_passes}_{jogador}.png',quality=95,facecolor='#2C2B2B')
          st.image(f'content/quadro_{lista_passes}_{jogador}.png')
          st.markdown(get_binary_file_downloader_html(f'content/quadro_{lista_passe}_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
      if 'Simples' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['events']=='Pass')&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['events']=='Pass')&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Infiltrado' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['TB']==1)&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['TB']==1)&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Chave' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['KP']==1)&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['KP']==1)&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)  
      if 'Cruzamento' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['events']=='cross')&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['events']=='cross')&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Assistência' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['Assist']==1)&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&(df_jogador['Assist']==1)&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Falta' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&((df_jogador['events']=='freekick_short')|(df_jogador['events']=='freekick_crossed'))&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&((df_jogador['events']=='freekick_short')|(df_jogador['events']=='freekick_crossed'))&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Escanteio' in lista_passes:
          passe_certo=df_jogador[(df_jogador['type_displayName']=='Pass')&((df_jogador['events']=='corner_short')|(df_jogador['events']=='corner_crossed'))&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=df_jogador[(df_jogador['type_displayName']=='Pass')&((df_jogador['events']=='corner_short')|(df_jogador['events']=='corner_crossed'))&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
      if 'Progressivo' in lista_passes:
          df_jogador=df_jogador[(df_jogador['type_displayName']=='Pass')].reset_index(drop=True)
          df_jogador['dist1'] = np.sqrt((105-df_jogador.x)**2 + (34-df_jogador.y)**2)
          df_jogador['dist2'] = np.sqrt((105-df_jogador.endX)**2 + (34-df_jogador.endY)**2)
          df_jogador['distdiff'] = df_jogador['dist1'] - df_jogador['dist2']
          pass1 = df_jogador.query("(x<52.5)&(endX<52.5)&(distdiff>=30)")
          pass2 = df_jogador.query("(x<52.5)&(endX>52.5)&(distdiff>=15)")
          pass3 = df_jogador.query("(x>52.5)&(endX>52.5)&(distdiff>=10)")
          pass1 = pass1.append(pass2)
          pass1 = pass1.append(pass3)
          passe_certo=pass1[(pass1['outcomeType_displayName']=='Successful')].reset_index(drop=True)
          passe_errado=pass1[(pass1['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
          passes(passe_certo,passe_errado)
   if grafico == 'Ações Defensivas':
      tipos_defesa=['Desarme','Interceptação','Corte','Bloqueio','Aéreo','Duelo']
      lista_defesa=st.multiselect('Escolha as ações',tipos_defesa)
      dct_defense={'Desarme':'Tackle','Interceptação':'Interception','Corte':'Clearance',
                   'Bloqueio':'BlockedPass','Aéreo':'Aerial','Duelo':'Challenge'}
      defesa=[dct_defense[k] for k in lista_defesa]
      defesa_certo=df_jogador[(df_jogador['type_displayName'].isin(defesa))&(df_jogador['outcomeType_displayName']=='Successful')].reset_index(drop=True)
      defesa_errado=df_jogador[(df_jogador['type_displayName'].isin(defesa))&(df_jogador['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
      cor_fundo = '#2c2b2b'
      fig, ax = plt.subplots(figsize=(15,10))
      pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                      stripe=False, line_zorder=2)
      pitch.draw(ax=ax)
      zo=12
      plt.scatter(data=defesa_certo, x='x',y='y',color='#00FF79',zorder=zo+1)
      plt.scatter(data=defesa_errado, x='x',y='y',color='#FD2B2C',zorder=zo+1)
      defensivo=df_jogador[(df_jogador['type_displayName'].isin(defesa))].reset_index(drop=True)
      plt.axvline(x=defensivo['x'].mean(),ymin=0.05, ymax=0.95, color='#7AB5B7', linestyle='--',lw=2)
      plt.savefig(f'content/defesa_{jogador}.png',dpi=300,facecolor=cor_fundo)
      im = Image.open(f'content/defesa_{jogador}.png')
      cor_fundo = '#2c2b2b'
      tamanho_arte = (3000, 2740)
      arte = Image.new('RGB',tamanho_arte,cor_fundo)
      W,H = arte.size
      im = im.rotate(90,expand=5)
      w,h= im.size
      im = im.resize((int(w/2),int(h/2)))
      im = im.copy()
      w,h= im.size
      arte.paste(im,(100,400))

      font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
      msg = f'{grafico}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,100),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{team}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,300),msg, fill='white',spacing= 20,font=font)

      acerto=len(defesa_certo)
      total=(len(defesa_certo)+len(defesa_errado))
      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{jogador}: {acerto}/{total}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,500),msg, fill='white',spacing= 20,font=font)

      im = Image.open('Arquivos/legenda-acerto-erro.png')
      w,h = im.size
      im = im.resize((int(w/5),int(h/5)))
      im = im.copy()
      arte.paste(im,(330,2350))

      font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
      msg = f'Certo'
      draw = ImageDraw.Draw(arte)
      draw.text((600,2400),msg, fill='white',spacing= 30,font=font)


      font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
      msg = f'Errado'
      draw = ImageDraw.Draw(arte)
      draw.text((920,2400),msg, fill='white',spacing= 30,font=font)

      fot =Image.open('Logos/Copy of pro_branco.png')
      w,h = fot.size
      fot = fot.resize((int(w/1.5),int(h/1.5)))
      fot = fot.copy()
      arte.paste(fot,(1870,1880),fot)

      times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
      logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
      try:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100),im)
      except:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100))

      arte.save(f'content/quadro_{grafico}_{jogador}.png',quality=95,facecolor='#2C2B2B')
      st.image(f'content/quadro_{grafico}_{jogador}.png')
      st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
   if grafico == 'Sonar Inverso de chutes':
      def sonarinverso(df):
        shots=df[df['events']=='Shot'].reset_index(drop=True)
        def sonarplotter(dataframe):
            df = dataframe[['x','y']].reset_index(drop=True)
            # df['Y'] = df['Y']*68
            # df['X'] = df['X']*105
            df['distance'] = np.sqrt((df['x']-105)**2+(df['y']-34)**2)
            df['angle'] = np.degrees(np.arctan2(105-df['x'],34-df['y'])) + 180

            angs = [180,200,220,240,260,280,300,320,340]
            rads = []
            density = []

            for angle in angs:
                angdf = df[(df.angle > angle)&(df.angle<=angle+20)]
                median_dist = angdf.distance.median()
                rads.append(median_dist)
                density.append(len(angdf))
            md = min(density)
            Md = max(density)
            density = [(i - md)/(Md - md) for i in density]

            return (angs, rads, density, df)

        cor_fundo = '#2c2b2b'
        fig, ax = plt.subplots(figsize=(15,10))
        pitch = VerticalPitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,half=True,
                        stripe=False, line_zorder=1)
        pitch.draw(ax=ax)

        from matplotlib.colors import ListedColormap, LinearSegmentedColormap

        cmaplist = [cor_fundo, '#F43B87']
        cmap = LinearSegmentedColormap.from_list("", cmaplist)



        angs, rads, cols, sdf = sonarplotter(shots)

        for j in range(9):
            wedge = mpatches.Wedge((34, 105), rads[j], angs[j], angs[j]+20, color = cmap(cols[j]),
                                      ec = '#f7e9ec')
            ax.add_patch(wedge)
        plt.savefig(f'content/sonar_{jogador}.png',dpi=300,facecolor=cor_fundo)
        im=Image.open(f'content/sonar_{jogador}.png')
        tamanho_arte = (3000, 2740)
        arte = Image.new('RGB',tamanho_arte,cor_fundo)
        W,H = arte.size
        w,h= im.size
        im = im.resize((int(w/1.5),int(h/1.5)))
        im = im.copy()
        arte.paste(im,(-250,700))

        font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
        msg = f'Sonar Inverso de Chutes'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((430,100),msg, fill='white',spacing= 20,font=font)

        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{team}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((430,350),msg, fill='white',spacing= 20,font=font)

        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{jogador}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((430,500),msg, fill='white',spacing= 20,font=font)

        ontarget=shots[~(shots['type_displayName']=='MissedShots')].reset_index(drop=True)
        target=len(ontarget)
        total = len(shots)
        gols=shots[shots['type_displayName']=='Goal'].reset_index(drop=True)
        if gols.empty == True:
          gols=0

        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'Chutes no alvo: {target} / {total}   |   Gols:  {gols} '
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((430,650),msg, fill='white',spacing= 20,font=font)

        fot =Image.open('Logos/Copy of pro_branco.png')
        w,h = fot.size
        fot = fot.resize((int(w/2),int(h/2)))
        fot = fot.copy()
        arte.paste(fot,(2350,2200),fot)

        times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
        logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
        try:
          r = requests.get(logo_url)
          im_bt = r.content
          image_file = io.BytesIO(im_bt)
          im = Image.open(image_file)
          w,h = im.size
          im = im.resize((int(w*2.5),int(h*2.5)))
          im = im.copy()
          arte.paste(im,(2500,100),im)
        except:
          r = requests.get(logo_url)
          im_bt = r.content
          image_file = io.BytesIO(im_bt)
          im = Image.open(image_file)
          w,h = im.size
          im = im.resize((int(w*2.5),int(h*2.5)))
          im = im.copy()
          arte.paste(im,(2500,100))

        font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
        msg = f'*Penâltis, cobranças de falta e gol contra não incluídos'
        draw = ImageDraw.Draw(arte)
        draw.text((430,2640),msg, fill='white',spacing= 30,font=font)
        arte.save(f'content/quadro_{grafico}_{jogador}.png',quality=95,facecolor='#2C2B2B')
        st.image(f'content/quadro_{grafico}_{jogador}.png')
        st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
      sonarinverso(df_jogador)
   if grafico == 'Passes mais frequentes':
      df_passe_plot= df_jogador[(df_jogador.events.isin(['Pass','cross']))&
         (df_jogador.outcomeType_displayName=='Successful')].reset_index(drop=True)
      df_passe_plot=df_passe_plot[['x','y','endX','endY']]
      valores = df_passe_plot.values
      passes_total = len(df_passe_plot)
      max_cluster = int(len(df_passe_plot)/8)
      min_cluster = int(len(df_passe_plot)/15)
      dic_geral_clusters = []
      df_passe = df_passe_plot

      dic_sill = {}
      for i in range(min_cluster, max_cluster):
        km = KMeans(n_clusters=i)
        km.fit(valores)
        label = km.predict(valores)
        sill = silhouette_score(valores,label)
        dic_sill.update({i:sill})

      df_sill = pd.DataFrame(dic_sill,index=[0]).transpose().reset_index()
      n_cluster = df_sill.sort_values(0,ascending=False).reset_index()['index'][0]
      valor =  df_sill.sort_values(0,ascending=False).reset_index()[0][0]
      print(f'{n_cluster}:{valor}')
      dic_geral_clusters.append({'metodo':'kmeans','n_cluster':'n_cluster','acerto':valor})

      km = KMeans(
          n_clusters=n_cluster, init='random',
          n_init=1, max_iter=300, 
          tol=1e-04, random_state=0
      )
      y_km = km.fit_predict(valores)

      # cor_fundo='#2C2B2B'
      df_passe['cluster'] = y_km
      df_passe['quantidade'] = 0
      cluster = df_passe.groupby('cluster')['quantidade'].count().reset_index().sort_values('quantidade',ascending=False).reset_index(drop=True)
      lista_cluster = list(cluster['cluster'])[0:3]
      df_plot = df_passe[df_passe['cluster'].isin(lista_cluster)].reset_index(drop=True)
      # df_plot
      cor_fundo = '#2c2b2b'
      fig, ax = plt.subplots(figsize=(15,10))
      pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                      stripe=False, line_zorder=1)
      pitch.draw(ax=ax)
      def plot_scatter_df(df,cor,zo):
        pitch.scatter(df.endX, df.endY, s=200, edgecolors=cor,lw=2, c=cor_fundo, zorder=zo+1, ax=ax)
        # plt.scatter(data=df, x='to_x',y='to_y',color=cor,zorder=zo+1,label='df',edgecolors='white',s=200)
        for linha in range(len(df)):
          x_inicial = df['x'][linha]
          y_inicial = df['y'][linha]
          x_final = df['endX'][linha]
          y_final = df['endY'][linha]
          lc1 = pitch.lines(x_inicial, y_inicial,
                        x_final, y_final,
                        lw=5, transparent=True, comet=True,
                        color=cor, ax=ax,zorder=zo)
      
      lista_cor = ['#FF4E63','#8D9713','#00A6FF']
      for clus,cor in zip(lista_cluster,lista_cor):
        df = (df_plot[df_plot['cluster'] == clus].reset_index())
        df['cor'] = cor 
        plot_scatter_df(df,cor,2)
      plt.show()
      plt.savefig(f'content/cluster_{jogador}.png',dpi=300,facecolor=cor_fundo)
      im = Image.open(f'content/cluster_{jogador}.png')
      cor_fundo = '#2c2b2b'
      tamanho_arte = (3000, 2740)
      arte = Image.new('RGB',tamanho_arte,cor_fundo)
      W,H = arte.size
      im = im.rotate(90,expand=5)
      w,h= im.size
      im = im.resize((int(w/2),int(h/2)))
      im = im.copy()
      w,h= im.size
      arte.paste(im,(100,400))

      font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
      msg = f'Cluster de passes'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,100),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{team}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,300),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{jogador}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,500),msg, fill='white',spacing= 20,font=font)

      fot =Image.open('Logos/Copy of pro_branco.png')
      w,h = fot.size
      fot = fot.resize((int(w/1.5),int(h/1.5)))
      fot = fot.copy()
      arte.paste(fot,(1870,1880),fot)

      times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
      logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
      try:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100),im)
      except:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100))

      arte.save(f'content/quadro_{grafico}_{jogador}.png',quality=95,facecolor='#2C2B2B')
      st.image(f'content/quadro_{grafico}_{jogador}.png')
      st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{jogador}.png', 'Imagem'), unsafe_allow_html=True)
if choice == 'Gráficos times (Partida)':
  st.subheader('Plote os gráficos do time em uma partida do campeonato')
  lista_temporada=['2020','2021','Euro 2021','Copa America 2021']
  temporada=st.selectbox('Selecione a temporada',lista_temporada)
  if temporada == '2020':   
     df = pd.read_csv('br2020.csv',encoding = "utf-8-sig")
  if temporada == '2021':
     df = pd.read_csv('br2021.csv',encoding = "utf-8-sig")
  if temporada == '2021':
     df = pd.read_csv('br2021.csv',encoding = "utf-8-sig")
  if temporada == 'Euro 2021':
     df = pd.read_csv('euro2021.csv',encoding = "utf-8-sig")
     teams_dict= {'Brazil':'Brasil','Paraguay':'Paraguai','Uruguay':'Uruguai','Colombia':'Colômbia','Ecuador':'Equador',
                   'Italy':'Itália','Switzerland':'Suíça','Turkey':'Turquia','Wales':'Gales','Belgium':'Bélgica','Denmark':'Dinamarca',
                   'Finland':'Finlândia','Russia':'Rússia','Netherlands':'Holanda','North Macedonia':'Macedônia do norte',
                   'Ukraine':'Ucrânia','Poland':'Polônia','Slovakia':'Eslováquia','Spain':'Espanha','Sweden':'Suécia',
                   'Croatia':'Croácia','Czech Republic':'Rep. Tcheca','England':'Inglaterra','Scotland':'Escócia',
                   'France':'França','Germany':'Alemanha','Hungary':'Hungria','Austria':'Áustria','Portugal':'Portugal'}
     df['hometeam']=df['hometeam'].map(teams_dict)
     df['awayteam']=df['awayteam'].map(teams_dict)
  if temporada == 'Copa America 2021':
     df = pd.read_csv('america2021.csv',encoding = "utf-8-sig")
     teams_dict= {'Brazil':'Brasil','Paraguay':'Paraguai','Uruguay':'Uruguai','Colombia':'Colômbia','Ecuador':'Equador',
                   'Italy':'Itália','Switzerland':'Suíça','Turkey':'Turquia','Wales':'Gales','Belgium':'Bélgica','Denmark':'Dinamarca',
                   'Finland':'Finlândia','Russia':'Rússia','Netherlands':'Holanda','North Macedonia':'Macedônia do norte',
                   'Ukraine':'Ucrânia','Poland':'Polônia','Slovakia':'Eslováquia','Spain':'Espanha','Sweden':'Suécia',
                   'Croatia':'Croácia','Czech Republic':'Rep. Tcheca','England':'Inglaterra','Scotland':'Escócia',
                   'France':'França','Germany':'Alemanha','Hungary':'Hungria'}
     df['hometeam']=df['hometeam'].map(teams_dict)
     df['awayteam']=df['awayteam'].map(teams_dict)
  nav1,nav2 = st.beta_columns(2)
  with nav1:
      home_team=st.selectbox('Time da casa',sorted(list(df['hometeam'].unique())))
  with nav2:
      away_team=st.selectbox('Time de fora',sorted(list(df['awayteam'].unique())))
  match=df[(df['hometeam']==home_team)&(df['awayteam']==away_team)].reset_index(drop=True)
  team=st.selectbox('Escolha o time',[home_team,away_team])
  df_team=match[((match['hometeam']==team) & (match['hometeamid']==match.teamId)) | ((match['awayteam']==team) & (match['awayteamid']==match.teamId))].reset_index(drop=True)
  lista_graficos=['Mapa de Passes','Posição Defensiva','Cruzamentos','Progressivos','Ações Defensivas','Passes mais frequentes',
                  'Entradas na Área','PPDA','Posse','Passes valiosos','Retomadas de Bola','Sonar Inverso de chutes']
  grafico=st.selectbox('Escolha o gráfico',lista_graficos)
  if grafico == 'Mapa de Passes':
    def mapa_de_passes(df):
      subs = df[df['type_displayName']=='SubstitutionOff']
      subs = subs['time_seconds']
      firstSub = subs.min()
      df = df[df['time_seconds'] < firstSub]
      df=df[(df.events.isin(['Pass','cross']))&
         (df.outcomeType_displayName=='Successful')].reset_index(drop=True)
      # passe_geral=df
      posicao_media_geral=df[['x','y']].groupby(by=df["shirtNo"]).mean().reset_index()
      titulares=df[['name',"shirtNo"]].drop_duplicates().dropna().reset_index(drop=True)
      pass_between = df.groupby(['name','receiver']).teamId.count().reset_index()
      pass_between.rename({'teamId':'to_success'},axis='columns',inplace=True)
      passe_geral=pass_between
      cor_fundo = '#2c2b2b'
      fig, ax = plt.subplots(figsize=(15,10))
      pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                    stripe=False, line_zorder=2)
      pitch.draw(ax=ax)
      pl_dict = dict(zip(df['shirtNo'], df['name']))
      posicao_media_geral['name']= posicao_media_geral['shirtNo'].map(pl_dict)
      posicao_media_geral=posicao_media_geral.set_index('name')
      posicao = posicao_media_geral
      cor_time = '#8c979b'
      for i in range(len(posicao)):
          w = posicao.iloc[i]['x']
          z = posicao.iloc[i]['y']
          ax.scatter(w,z, color=cor_time, s=800,zorder=3)
          ax.annotate(str(int(posicao.shirtNo[i])),xy= (w,z),rotation=-90,va='center',ha='center',fontsize=18) 

      for linha in range(len(passe_geral)):
        passe = (passe_geral['name'][linha])
        recebeu = (passe_geral['receiver'][linha])
        quantidade_passes = (passe_geral['to_success'][linha])
        distancia_ponto = 3
          

        x_inicial = (posicao.loc['{}'.format(passe),'x'])
        x_final = (posicao.loc['{}'.format(recebeu),'x'])
        y_inicial = (posicao.loc['{}'.format(passe),'y'])
        y_final = (posicao.loc['{}'.format(recebeu),'y'])

        frente = (x_final > x_inicial)
        tras =  (x_final < x_inicial)
        esquerda = (y_final < y_inicial)
        direita = (y_final > y_inicial)


        curto_frente = ((x_final - x_inicial) < 9 and (x_final - x_inicial) > 0 )
        curto_tras =  ((x_inicial - x_final) < 9 and (x_inicial - x_final) > 0)
        curto_direita = ((y_final - y_inicial) < 9 and (y_final - y_inicial) > 0)
        curto_esquerda = ((y_inicial - y_final) < 9 and (y_inicial - y_final) > 0) 
          

        cor_ponto = 'white'
        linha_forte = 2.5
        cor_linha = '#00FFD2'
        linha_fraca = 0.5
        s=10
        
        if quantidade_passes > 8:
          if True:
            if curto_frente and curto_direita:
              plt.scatter(x_final - 1.5, y_final - 3, s=s, color=cor_ponto,zorder=3)
              plt.plot([x_inicial + 1.5, x_final - 1.5], [y_inicial + 1, y_final - 3], color=cor_linha, linewidth=linha_forte)
            elif curto_frente and direita:
              plt.scatter(x_final -1, y_final -3,s=s, color=cor_ponto,zorder=3)
              plt.plot([x_inicial -1 ,x_final -1 ],[y_inicial +3 ,y_final -3 ], color=cor_linha, linewidth=linha_forte)
            elif curto_direita and frente:
              plt.scatter(x_final -3, y_final -1.5  ,s=s, color=cor_ponto,zorder=3)
              plt.plot([x_inicial +3 ,x_final -3 ],[y_inicial  ,y_final -1.5  ], color=cor_linha, linewidth=linha_forte)
            else:
              if frente and direita:
                plt.scatter(x_final -2   , y_final -2   ,s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial +2   ,x_final -2  ],[y_inicial +2   ,y_final -2 ], color=cor_linha, linewidth=linha_forte) 

          if True:
            if curto_tras and curto_esquerda:
                plt.scatter(x_final + 1.5, y_final + 3, s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial - 1.5, x_final + 1.5], [y_inicial - 1, y_final + 3], color=cor_linha, linewidth=linha_forte)
            elif curto_tras and esquerda:
                plt.scatter(x_final +0.5 , y_final +3,s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial +0.5  ,x_final +0.5  ],[y_inicial -3 ,y_final +3 ], color=cor_linha, linewidth=linha_forte)
            elif curto_esquerda and tras:
                plt.scatter(x_final +3 , y_final+1.5   ,s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial -3 ,x_final +3 ],[y_inicial  ,y_final +1.5  ], color=cor_linha, linewidth=linha_forte)
            else:
                if tras and esquerda: 
                    plt.scatter(x_final +2   , y_final +2   ,s=s, color=cor_ponto,zorder=3)
                    plt.plot([x_inicial -2   ,x_final +2  ],[y_inicial -2   ,y_final +2 ], color=cor_linha, linewidth=linha_forte)
          if True:
            if curto_frente and curto_esquerda:
                plt.scatter(x_final, y_final, s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial, x_final], [y_inicial, y_final], color=cor_linha, linewidth=linha_forte)
            elif curto_frente and esquerda:
                plt.scatter(x_final, y_final +3 ,s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial +1 ,x_final  ],[y_inicial -3 ,y_final +3 ], color=cor_linha, linewidth=linha_forte)
            elif curto_esquerda and frente:
                plt.scatter(x_final -3 , y_final +1.5   ,s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial +3 ,x_final -3 ],[y_inicial +1.5  ,y_final +1.5  ], color=cor_linha, linewidth=linha_forte)
            else:
                if  frente and esquerda: 
                  plt.scatter(x_final -1.5  , y_final +3  ,s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial +2   ,x_final -1.5   ],[y_inicial +1,y_final +3 ], color=cor_linha, linewidth=linha_forte)
          
          if True:
            if curto_tras and curto_direita:
                plt.scatter(x_final, y_final, s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial, x_final], [y_inicial, y_final], color=cor_linha, linewidth=linha_forte)
            elif curto_tras and direita:
                plt.scatter(x_final, y_final -3 ,s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial -1  ,x_final  ],[y_inicial +3 ,y_final -3 ], color=cor_linha, linewidth=linha_forte)
            elif curto_direita and tras:
                plt.scatter(x_final +3 , y_final -1.5   ,s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial -3 ,x_final +3 ],[y_inicial -1.5  ,y_final -1.5  ], color=cor_linha, linewidth=linha_forte)
            else:
                if tras and direita: 
                    plt.scatter(x_final + 1.5, y_final - 3, s=s, color=cor_ponto,zorder=3)
                    plt.plot([x_inicial - 2, x_final + 1.5], [y_inicial - 1, y_final - 3], color=cor_linha, linewidth=linha_forte)


        if quantidade_passes > 3 : 
          if True:
              if curto_frente and curto_direita:
                plt.scatter(x_final - 1.5, y_final - 3, s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial + 1.5, x_final - 1.5], [y_inicial + 1, y_final - 3], color=cor_linha, linewidth=linha_fraca)
              elif curto_frente and direita:
                plt.scatter(x_final -1, y_final -3,s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial -1 ,x_final -1 ],[y_inicial +3 ,y_final -3 ], color=cor_linha, linewidth=linha_fraca)
              elif curto_direita and frente:
                plt.scatter(x_final -3, y_final -1.5  ,s=s, color=cor_ponto,zorder=3)
                plt.plot([x_inicial +3 ,x_final -3 ],[y_inicial  ,y_final -1.5  ], color=cor_linha, linewidth=linha_fraca)
              else:
                if frente and direita:
                  plt.scatter(x_final -2   , y_final -2   ,s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial +2   ,x_final -2  ],[y_inicial +2   ,y_final -2 ], color=cor_linha, linewidth=linha_fraca) 

          if True:
              if curto_tras and curto_esquerda:
                  plt.scatter(x_final + 1.5, y_final + 3, s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial - 1.5, x_final + 1.5], [y_inicial - 1, y_final + 3], color=cor_linha, linewidth=linha_fraca)
              elif curto_tras and esquerda:
                  plt.scatter(x_final +0.5 , y_final +3,s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial +0.5  ,x_final +0.5  ],[y_inicial -3 ,y_final +3 ], color=cor_linha, linewidth=linha_fraca)
              elif curto_esquerda and tras:
                  plt.scatter(x_final +3 , y_final+1.5   ,s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial -3 ,x_final +3 ],[y_inicial  ,y_final +1.5  ], color=cor_linha, linewidth=linha_fraca)
              else:
                  if tras and esquerda: 
                      plt.scatter(x_final +2   , y_final +2   ,s=s, color=cor_ponto,zorder=3)
                      plt.plot([x_inicial -2   ,x_final +2  ],[y_inicial -2   ,y_final +2 ], color=cor_linha, linewidth=linha_fraca)
          if True:
              if curto_frente and curto_esquerda:
                  plt.scatter(x_final, y_final, s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial, x_final], [y_inicial, y_final], color=cor_linha, linewidth=linha_fraca)
              elif curto_frente and esquerda:
                  plt.scatter(x_final, y_final +3 ,s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial +1 ,x_final  ],[y_inicial -3 ,y_final +3 ], color=cor_linha, linewidth=linha_fraca)
              elif curto_esquerda and frente:
                  plt.scatter(x_final -3 , y_final +1.5   ,s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial +3 ,x_final -3 ],[y_inicial +1.5  ,y_final +1.5  ], color=cor_linha, linewidth=linha_fraca)
              else:
                  if  frente and esquerda: 
                    plt.scatter(x_final -1.5  , y_final +3  ,s=s, color=cor_ponto,zorder=3)
                    plt.plot([x_inicial +2   ,x_final -1.5   ],[y_inicial +1,y_final +3 ], color=cor_linha, linewidth=linha_fraca)
            
          if True:
              if curto_tras and curto_direita:
                  plt.scatter(x_final, y_final, s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial, x_final], [y_inicial, y_final], color=cor_linha, linewidth=linha_fraca)
              elif curto_tras and direita:
                  plt.scatter(x_final, y_final -3 ,s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial -1  ,x_final  ],[y_inicial +3 ,y_final -3 ], color=cor_linha, linewidth=linha_fraca)
              elif curto_direita and tras:
                  plt.scatter(x_final +3 , y_final -1.5   ,s=s, color=cor_ponto,zorder=3)
                  plt.plot([x_inicial -3 ,x_final +3 ],[y_inicial -1.5  ,y_final -1.5  ], color=cor_linha, linewidth=linha_fraca)
              else:
                  if tras and direita: 
                      plt.scatter(x_final + 1.5, y_final - 3, s=s, color=cor_ponto,zorder=3)
                      plt.plot([x_inicial - 2, x_final + 1.5], [y_inicial - 1, y_final - 3], color=cor_linha, linewidth=linha_fraca)
      plt.show()
      plt.savefig(f'content/{grafico}_{team}.png',dpi=300,facecolor=cor_fundo)
      im = Image.open(f'content/{grafico}_{team}.png')
      cor_fundo = '#2c2b2b'
      tamanho_arte = (3000, 2740)
      arte = Image.new('RGB',tamanho_arte,cor_fundo)
      W,H = arte.size
      im = im.rotate(90,expand=5)
      w,h= im.size
      im = im.resize((int(w/1.75),int(h/1.75)))
      im = im.copy()
      w,h= im.size
      arte.paste(im,(50,100))

      font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
      msg = f'{grafico}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,100),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{home_team}- {away_team}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,300),msg, fill='white',spacing= 20,font=font)

      fot =Image.open('Logos/Copy of pro_branco.png')
      w,h = fot.size
      fot = fot.resize((int(w/1.5),int(h/1.5)))
      fot = fot.copy()
      arte.paste(fot,(1870,2180),fot)

      times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
      logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
      try:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100),im)
      except:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100))

      altura = 350
      for linha in range(len(titulares)):
        altura += 100
        nome = titulares['name'][linha]
        numero = int(titulares['shirtNo'][linha])



        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{numero}-{nome}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text(((1930,altura)),msg, fill='white',spacing= 20,font=font)



      font = ImageFont.truetype('Camber/Camber-Rg.ttf',50)
      msg = f'*Linhas fortes para mais de 8 passes'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text(((330,2350)),msg, fill='white',spacing= 20,font=font)
      font = ImageFont.truetype('Camber/Camber-Rg.ttf',50)
      msg = f'*Mínimo de 3 passes para plotar as linhas'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text(((330,2500)),msg, fill='white',spacing= 20,font=font)
      font = ImageFont.truetype('Camber/Camber-Rg.ttf',50)
      msg = f'*Mapa de passes feitos até a primeira substituição'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text(((330,2650)),msg, fill='white',spacing= 20,font=font)



      arte.save(f'content/quadro_{grafico}_{team}.png',quality=95,facecolor='#2C2B2B')
      st.image(f'content/quadro_{grafico}_{team}.png')
      st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{team}.png', 'Imagem'), unsafe_allow_html=True)
    mapa_de_passes(df_team)
  if grafico == 'Posição Defensiva':
    def pos_defensiva(df):
      subs = df[df['type_displayName']=='SubstitutionOff']
      subs = subs['time_seconds']
      firstSub = subs.min()
      df = df[df['time_seconds'] < firstSub]
      lista_def=['Tackle','Interception','Clearance','Blocked Pass','Challenge','Aerial']
      defesa=df[((df['type_displayName'].isin(lista_def))&(df['outcomeType_displayName']=='Successful')) | ((df['type_displayName']=='Foul')&(df['outcomeType_displayName']=='Unsuccessful'))].reset_index(drop=True)
      defesa=defesa[~(defesa['position']=='GK')]
      defesa=defesa[['x','y']].groupby(by=defesa["shirtNo"]).mean().reset_index()
      titulares=df[['name',"shirtNo"]].drop_duplicates().dropna().reset_index(drop=True)
      cor_fundo = '#2c2b2b'
      fig, ax = plt.subplots(figsize=(15,10))
      pitch = VerticalPitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                      stripe=False, line_zorder=2)
      pitch.draw(ax=ax)
      from scipy.spatial import ConvexHull
      for i in range(len(defesa)):
        x = defesa['x'][i]
        y = defesa['y'][i]

        plt.scatter(y,x,color='white',zorder=3,s=500)
        ax.annotate(str(int(defesa.shirtNo[i])),xy= (y,x) ,rotation=0,size=10,va='center',ha='center')

      def encircle(y,x, ax=None, **kw):
        if not ax: ax=plt.gca()
        p = np.c_[y,x]
        hull = ConvexHull(p)
        poly = plt.Polygon(p[hull.vertices,:], **kw)
        ax.add_patch(poly)

      encircle(defesa['y'], defesa['x'], ec="k", fc="pink", alpha=0.7)
      plt.show()
      plt.savefig(f'content/{grafico}_{team}.png',dpi=300,facecolor=cor_fundo)
      im = Image.open(f'content/{grafico}_{team}.png')
      cor_fundo = '#2c2b2b'
      tamanho_arte = (3000, 2740)
      arte = Image.new('RGB',tamanho_arte,cor_fundo)
      W,H = arte.size
      im = im.rotate(0,expand=5)
      w,h= im.size
      im = im.resize((int(w/1.25),int(h/1.25)))
      im = im.copy()
      w,h= im.size
      arte.paste(im,(-960,240))

      font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
      msg = f'{grafico}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,100),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{home_team}- {away_team}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,400),msg, fill='white',spacing= 20,font=font)

      fot =Image.open('Logos/Copy of pro_branco.png')
      w,h = fot.size
      fot = fot.resize((int(w/1.5),int(h/1.5)))
      fot = fot.copy()
      arte.paste(fot,(1870,1880),fot)

      times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
      logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
      try:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100),im)
      except:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100))

      altura = 500
      for linha in range(len(titulares)):
        altura += 100
        nome = titulares['name'][linha]
        numero = int(titulares['shirtNo'][linha])



        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{numero}-{nome}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text(((1930,altura)),msg, fill='white',spacing= 20,font=font)



      font = ImageFont.truetype('Camber/Camber-Rg.ttf',50)
      msg = f'*Os pontos correspondem a posição média dos jogadores, baseado em suas ações defensivas'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text(((330,2500)),msg, fill='white',spacing= 20,font=font)
      arte.save(f'content/quadro_{grafico}_{team}.png',quality=95,facecolor='#2C2B2B')
      st.image(f'content/quadro_{grafico}_{team}.png')
      st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{team}.png', 'Imagem'), unsafe_allow_html=True)
    pos_defensiva(df_team)
  if grafico == 'Cruzamentos':
    def passes(df1,df2):
        cor_fundo = '#2c2b2b'
        fig, ax = plt.subplots(figsize=(15,10))
        pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                        stripe=False, line_zorder=2)
        pitch.draw(ax=ax)
        certo=df1
        errado=df2
        def plot_scatter_df(df,cor,zo):
            pitch.scatter(df.endX, df.endY, s=200, edgecolors=cor,lw=2, c=cor_fundo, zorder=zo+1, ax=ax)
            # plt.scatter(data=df, x='to_x',y='to_y',color=cor,zorder=zo+1,label='df',edgecolors='white',s=200):
            x_inicial = df['x']
            y_inicial = df['y']
            x_final = df['endX']
            y_final = df['endY']
            lc1 = pitch.lines(x_inicial, y_inicial,
                          x_final,y_final,
                          lw=5, transparent=True, comet=True,color=cor, ax=ax,zorder=zo)
        plot_scatter_df(certo,'#00FF79',12)
        plot_scatter_df(errado,'#FD2B2C',9)
        plt.show()
        plt.savefig(f'content/passe_{team}.png',dpi=300,facecolor=cor_fundo)
        im = Image.open(f'content/passe_{team}.png')
        cor_fundo = '#2c2b2b'
        tamanho_arte = (3000, 2740)
        arte = Image.new('RGB',tamanho_arte,cor_fundo)
        W,H = arte.size
        im = im.rotate(90,expand=5)
        w,h= im.size
        im = im.resize((int(w/2),int(h/2)))
        im = im.copy()
        w,h= im.size
        arte.paste(im,(100,400))

        font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
        msg = f'{grafico}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,100),msg, fill='white',spacing= 20,font=font)

        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{home_team}- {away_team}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,300),msg, fill='white',spacing= 20,font=font)

        acerto=len(passe_certo)
        total=(len(passe_certo)+len(passe_errado))
        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{team}: {acerto}/{total}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,500),msg, fill='white',spacing= 20,font=font)

        im = Image.open('Arquivos/legenda-acerto-erro.png')
        w,h = im.size
        im = im.resize((int(w/5),int(h/5)))
        im = im.copy()
        arte.paste(im,(330,2350))

        font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
        msg = f'Certo'
        draw = ImageDraw.Draw(arte)
        draw.text((600,2400),msg, fill='white',spacing= 30,font=font)


        font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
        msg = f'Errado'
        draw = ImageDraw.Draw(arte)
        draw.text((920,2400),msg, fill='white',spacing= 30,font=font)

        fot =Image.open('Logos/Copy of pro_branco.png')
        w,h = fot.size
        fot = fot.resize((int(w/1.5),int(h/1.5)))
        fot = fot.copy()
        arte.paste(fot,(1870,1880),fot)

        times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
        logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
        try:
          r = requests.get(logo_url)
          im_bt = r.content
          image_file = io.BytesIO(im_bt)
          im = Image.open(image_file)
          w,h = im.size
          im = im.resize((int(w*2.5),int(h*2.5)))
          im = im.copy()
          arte.paste(im,(2500,100),im)
        except:
          r = requests.get(logo_url)
          im_bt = r.content
          image_file = io.BytesIO(im_bt)
          im = Image.open(image_file)
          w,h = im.size
          im = im.resize((int(w*2.5),int(h*2.5)))
          im = im.copy()
          arte.paste(im,(2500,100))

        arte.save(f'content/quadro_{grafico}_{team}.png',quality=95,facecolor='#2C2B2B')
        st.image(f'content/quadro_{grafico}_{team}.png')
        st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{team}.png', 'Imagem'), unsafe_allow_html=True)
    passe_certo=df_team[(df_team['type_displayName']=='Pass')&(df_team['events']=='cross')&(df_team['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    passe_errado=df_team[(df_team['type_displayName']=='Pass')&(df_team['events']=='cross')&(df_team['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
    passes(passe_certo,passe_errado)
  if grafico == 'Progressivos':
    def passes(df1,df2):
        cor_fundo = '#2c2b2b'
        fig, ax = plt.subplots(figsize=(15,10))
        pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                        stripe=False, line_zorder=2)
        pitch.draw(ax=ax)
        certo=df1
        errado=df2
        def plot_scatter_df(df,cor,zo):
            pitch.scatter(df.endX, df.endY, s=200, edgecolors=cor,lw=2, c=cor_fundo, zorder=zo+1, ax=ax)
            # plt.scatter(data=df, x='to_x',y='to_y',color=cor,zorder=zo+1,label='df',edgecolors='white',s=200):
            x_inicial = df['x']
            y_inicial = df['y']
            x_final = df['endX']
            y_final = df['endY']
            lc1 = pitch.lines(x_inicial, y_inicial,
                          x_final,y_final,
                          lw=5, transparent=True, comet=True,color=cor, ax=ax,zorder=zo)
        plot_scatter_df(certo,'#00FF79',12)
        plot_scatter_df(errado,'#FD2B2C',9)
        plt.show()
        plt.savefig(f'content/passe_{team}.png',dpi=300,facecolor=cor_fundo)
        im = Image.open(f'content/passe_{team}.png')
        cor_fundo = '#2c2b2b'
        tamanho_arte = (3000, 2740)
        arte = Image.new('RGB',tamanho_arte,cor_fundo)
        W,H = arte.size
        im = im.rotate(90,expand=5)
        w,h= im.size
        im = im.resize((int(w/2),int(h/2)))
        im = im.copy()
        w,h= im.size
        arte.paste(im,(100,400))

        font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
        msg = f'{grafico}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,100),msg, fill='white',spacing= 20,font=font)

        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{home_team}- {away_team}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,300),msg, fill='white',spacing= 20,font=font)

        acerto=len(passe_certo)
        total=(len(passe_certo)+len(passe_errado))
        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{team}: {acerto}/{total}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,500),msg, fill='white',spacing= 20,font=font)

        im = Image.open('Arquivos/legenda-acerto-erro.png')
        w,h = im.size
        im = im.resize((int(w/5),int(h/5)))
        im = im.copy()
        arte.paste(im,(330,2350))

        font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
        msg = f'Certo'
        draw = ImageDraw.Draw(arte)
        draw.text((600,2400),msg, fill='white',spacing= 30,font=font)


        font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
        msg = f'Errado'
        draw = ImageDraw.Draw(arte)
        draw.text((920,2400),msg, fill='white',spacing= 30,font=font)

        fot =Image.open('Logos/Copy of pro_branco.png')
        w,h = fot.size
        fot = fot.resize((int(w/1.5),int(h/1.5)))
        fot = fot.copy()
        arte.paste(fot,(1870,1880),fot)

        times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
        logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
        try:
          r = requests.get(logo_url)
          im_bt = r.content
          image_file = io.BytesIO(im_bt)
          im = Image.open(image_file)
          w,h = im.size
          im = im.resize((int(w*2.5),int(h*2.5)))
          im = im.copy()
          arte.paste(im,(2500,100),im)
        except:
          r = requests.get(logo_url)
          im_bt = r.content
          image_file = io.BytesIO(im_bt)
          im = Image.open(image_file)
          w,h = im.size
          im = im.resize((int(w*2.5),int(h*2.5)))
          im = im.copy()
          arte.paste(im,(2500,100))

        arte.save(f'content/quadro_{grafico}_{team}.png',quality=95,facecolor='#2C2B2B')
        st.image(f'content/quadro_{grafico}_{team}.png')
        st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{team}.png', 'Imagem'), unsafe_allow_html=True)
    df_team=df_team[(df_team['type_displayName']=='Pass')].reset_index(drop=True)
    df_team['dist1'] = np.sqrt((105-df_team.x)**2 + (34-df_team.y)**2)
    df_team['dist2'] = np.sqrt((105-df_team.endX)**2 + (34-df_team.endY)**2)
    df_team['distdiff'] = df_team['dist1'] - df_team['dist2']
    pass1 = df_team.query("(x<52.5)&(endX<52.5)&(distdiff>=30)")
    pass2 = df_team.query("(x<52.5)&(endX>52.5)&(distdiff>=15)")
    pass3 = df_team.query("(x>52.5)&(endX>52.5)&(distdiff>=10)")
    pass1 = pass1.append(pass2)
    pass1 = pass1.append(pass3)
    passe_certo=pass1[(pass1['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    passe_errado=pass1[(pass1['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
    passes(passe_certo,passe_errado)
  if grafico == 'Entradas na Área':
    def box_entry(df):
      passdf = df[(df.events.isin(['Pass','cross']))&
         (df.outcomeType_displayName=='Successful')].reset_index(drop=True)
      insideboxendx = passdf.endX>=88.5
      insideboxstartx = passdf.x>=88.5
      outsideboxstartx = passdf.x<88.5
      insideboxendy = np.abs(34-passdf.endY)<20.16
      insideboxstarty = np.abs(34-passdf.y)<20.16
      allbox = passdf[~(insideboxstartx&insideboxstarty)]
      allbox = allbox[insideboxendx&insideboxendy]
      fig, ax = plt.subplots(figsize=(15,10))
      cor_fundo = '#2c2b2b'
      pitch = VerticalPitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                      stripe=False, line_zorder=2,half=False)
      pitch.draw(ax=ax)
      fig.set_facecolor(cor_fundo)
      from matplotlib.colors import LinearSegmentedColormap
      import matplotlib.patheffects as path_effects
      path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
      cmap = LinearSegmentedColormap.from_list('name', [cor_fundo, '#F43B87'])
      bin_statistic = pitch.bin_statistic_positional(allbox.x, allbox.y, statistic='count',
                                                    positional='full', normalize=True)
      pitch.heatmap_positional(bin_statistic, ax=ax,
                              cmap=cmap, edgecolors='#22312b')
      labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                                  ax=ax, ha='center', va='center',
                                  str_format='{:.0%}',path_effects=path_eff)
      plt.show()
      plt.savefig(f'content/passe_{team}.png',dpi=300,facecolor=cor_fundo)
      im = Image.open(f'content/passe_{team}.png')
      tamanho_arte = (3000, 2740)
      arte = Image.new('RGB',tamanho_arte,cor_fundo)
      W,H = arte.size
      im = im.rotate(0,expand=5)
      w,h= im.size
      im = im.resize((int(w/1.25),int(h/1.25)))
      im = im.copy()
      w,h= im.size
      arte.paste(im,(-970,310))

      font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
      msg = f'{grafico}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,100),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{home_team}- {away_team}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,300),msg, fill='white',spacing= 20,font=font)

      
      total=(len(allbox))
      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{team}: {total}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,500),msg, fill='white',spacing= 20,font=font)


      font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
      msg = f'*Apenas considerando passes simples e cruzamentos'
      draw = ImageDraw.Draw(arte)
      draw.text((330,2450),msg, fill='white',spacing= 30,font=font)

      fot =Image.open('Logos/Copy of pro_branco.png')
      w,h = fot.size
      fot = fot.resize((int(w/1.5),int(h/1.5)))
      fot = fot.copy()
      arte.paste(fot,(1870,1880),fot)

      times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
      logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
      try:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100),im)
      except:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100))

      arte.save(f'content/quadro_{grafico}_{team}.png',quality=95,facecolor='#2C2B2B')
      st.image(f'content/quadro_{grafico}_{team}.png')
      st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{team}.png', 'Imagem'), unsafe_allow_html=True)
    box_entry(df_team)
  if grafico == 'Retomadas de Bola':
    def recovery(df):
      recover = df[(df.events.isin(['BallRecovery']))&
         (df.outcomeType_displayName=='Successful')].reset_index(drop=True)
      fig, ax = plt.subplots(figsize=(15,10))
      cor_fundo = '#2c2b2b'
      pitch = VerticalPitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                      stripe=False, line_zorder=2,half=False)
      pitch.draw(ax=ax)
      fig.set_facecolor(cor_fundo)
      from matplotlib.colors import LinearSegmentedColormap
      import matplotlib.patheffects as path_effects
      path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
      cmap = LinearSegmentedColormap.from_list('name', [cor_fundo, '#F43B87'])
      bin_statistic = pitch.bin_statistic_positional(recover.x, recover.y, statistic='count',
                                                    positional='vertical', normalize=True)
      pitch.heatmap_positional(bin_statistic, ax=ax,
                              cmap=cmap, edgecolors='#22312b')
      labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                                  ax=ax, ha='center', va='center',
                                  str_format='{:.0%}',path_effects=path_eff)
      plt.show()
      plt.savefig(f'content/passe_{team}.png',dpi=300,facecolor=cor_fundo)
      im = Image.open(f'content/passe_{team}.png')
      tamanho_arte = (3000, 2740)
      arte = Image.new('RGB',tamanho_arte,cor_fundo)
      W,H = arte.size
      im = im.rotate(0,expand=5)
      w,h= im.size
      im = im.resize((int(w/1.25),int(h/1.25)))
      im = im.copy()
      w,h= im.size
      arte.paste(im,(-970,310))

      font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
      msg = f'{grafico}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,100),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{home_team}- {away_team}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,300),msg, fill='white',spacing= 20,font=font)

      
      total=(len(recover))
      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{team}: {total}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((330,500),msg, fill='white',spacing= 20,font=font)


      fot =Image.open('Logos/Copy of pro_branco.png')
      w,h = fot.size
      fot = fot.resize((int(w/1.5),int(h/1.5)))
      fot = fot.copy()
      arte.paste(fot,(1870,1880),fot)

      times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
      logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
      try:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100),im)
      except:
        r = requests.get(logo_url)
        im_bt = r.content
        image_file = io.BytesIO(im_bt)
        im = Image.open(image_file)
        w,h = im.size
        im = im.resize((int(w*2.5),int(h*2.5)))
        im = im.copy()
        arte.paste(im,(2500,100))

      arte.save(f'content/quadro_{grafico}_{team}.png',quality=95,facecolor='#2C2B2B')
      st.image(f'content/quadro_{grafico}_{team}.png')
      st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{team}.png', 'Imagem'), unsafe_allow_html=True)
    recovery(df_team)
  if grafico == 'Ações Defensivas':
    tipos_defesa=['Desarme','Interceptação','Corte','Bloqueio','Aéreo','Duelo']
    lista_defesa=st.multiselect('Escolha as ações',tipos_defesa)
    dct_defense={'Desarme':'Tackle','Interceptação':'Interception','Corte':'Clearance',
                  'Bloqueio':'BlockedPass','Aéreo':'Aerial','Duelo':'Challenge'}
    defesa=[dct_defense[k] for k in lista_defesa]
    defesa_certo=df_team[(df_team['type_displayName'].isin(defesa))&(df_team['outcomeType_displayName']=='Successful')].reset_index(drop=True)
    defesa_errado=df_team[(df_team['type_displayName'].isin(defesa))&(df_team['outcomeType_displayName']=='Unsuccessful')].reset_index(drop=True)
    cor_fundo = '#2c2b2b'
    fig, ax = plt.subplots(figsize=(15,10))
    pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                    stripe=False, line_zorder=2)
    pitch.draw(ax=ax)
    zo=12
    plt.scatter(data=defesa_certo, x='x',y='y',color='#00FF79',zorder=zo+1)
    plt.scatter(data=defesa_errado, x='y',y='x',color='#FD2B2C',zorder=zo+1)
    defensivo=df_team[(df_team['type_displayName'].isin(defesa))].reset_index(drop=True)
    plt.axvline(x=defensivo['x'].mean(),ymin=0.05, ymax=0.95, color='#7AB5B7', linestyle='--',lw=2)
    plt.show()
    plt.savefig(f'content/defesa_{team}.png',dpi=300,facecolor=cor_fundo)
    im = Image.open(f'content/defesa_{team}.png')
    cor_fundo = '#2c2b2b'
    tamanho_arte = (3000, 2740)
    arte = Image.new('RGB',tamanho_arte,cor_fundo)
    W,H = arte.size
    im = im.rotate(90,expand=5)
    w,h= im.size
    im = im.resize((int(w/2),int(h/2)))
    im = im.copy()
    w,h= im.size
    arte.paste(im,(100,400))

    font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
    msg = f'{grafico}'
    draw = ImageDraw.Draw(arte)
    w, h = draw.textsize(msg,spacing=20,font=font)
    draw.text((330,100),msg, fill='white',spacing= 20,font=font)

    font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
    msg = f'{home_team}- {away_team}'
    draw = ImageDraw.Draw(arte)
    w, h = draw.textsize(msg,spacing=20,font=font)
    draw.text((330,300),msg, fill='white',spacing= 20,font=font)

    acerto=len(defesa_certo)
    total=(len(defesa_certo)+len(defesa_errado))
    font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
    msg = f'{team}: {acerto}/{total}'
    draw = ImageDraw.Draw(arte)
    w, h = draw.textsize(msg,spacing=20,font=font)
    draw.text((330,500),msg, fill='white',spacing= 20,font=font)

    im = Image.open('Arquivos/legenda-acerto-erro.png')
    w,h = im.size
    im = im.resize((int(w/5),int(h/5)))
    im = im.copy()
    arte.paste(im,(330,2350))

    font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
    msg = f'Certo'
    draw = ImageDraw.Draw(arte)
    draw.text((600,2400),msg, fill='white',spacing= 30,font=font)


    font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
    msg = f'Errado'
    draw = ImageDraw.Draw(arte)
    draw.text((920,2400),msg, fill='white',spacing= 30,font=font)

    fot =Image.open('Logos/Copy of pro_branco.png')
    w,h = fot.size
    fot = fot.resize((int(w/1.5),int(h/1.5)))
    fot = fot.copy()
    arte.paste(fot,(1870,1880),fot)

    times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
    logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
    try:
      r = requests.get(logo_url)
      im_bt = r.content
      image_file = io.BytesIO(im_bt)
      im = Image.open(image_file)
      w,h = im.size
      im = im.resize((int(w*2.5),int(h*2.5)))
      im = im.copy()
      arte.paste(im,(2500,100),im)
    except:
      r = requests.get(logo_url)
      im_bt = r.content
      image_file = io.BytesIO(im_bt)
      im = Image.open(image_file)
      w,h = im.size
      im = im.resize((int(w*2.5),int(h*2.5)))
      im = im.copy()
      arte.paste(im,(2500,100))

    arte.save(f'content/quadro_{grafico}_{team}.png',quality=95,facecolor='#2C2B2B')
    st.image(f'content/quadro_{grafico}_{team}.png')
    st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{team}.png', 'Imagem'), unsafe_allow_html=True)
  if grafico == 'Passes mais frequentes':
    df_passe_plot= df_team[(df_team.events.isin(['Pass','cross']))&
         (df_team.outcomeType_displayName=='Successful')].reset_index(drop=True)
    df_passe_plot=df_passe_plot[['x','y','endX','endY']]
    valores = df_passe_plot.values
    passes_total = len(df_passe_plot)
    max_cluster = int(len(df_passe_plot)/8)
    min_cluster = int(len(df_passe_plot)/15)
    dic_geral_clusters = []
    df_passe = df_passe_plot

    dic_sill = {}
    for i in range(min_cluster, max_cluster):
      km = KMeans(n_clusters=i)
      km.fit(valores)
      label = km.predict(valores)
      sill = silhouette_score(valores,label)
      dic_sill.update({i:sill})

    df_sill = pd.DataFrame(dic_sill,index=[0]).transpose().reset_index()
    n_cluster = df_sill.sort_values(0,ascending=False).reset_index()['index'][0]
    valor =  df_sill.sort_values(0,ascending=False).reset_index()[0][0]
    print(f'{n_cluster}:{valor}')
    dic_geral_clusters.append({'metodo':'kmeans','n_cluster':'n_cluster','acerto':valor})

    km = KMeans(
        n_clusters=n_cluster, init='random',
        n_init=1, max_iter=300, 
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(valores)

    # cor_fundo='#2C2B2B'
    df_passe['cluster'] = y_km
    df_passe['quantidade'] = 0
    cluster = df_passe.groupby('cluster')['quantidade'].count().reset_index().sort_values('quantidade',ascending=False).reset_index(drop=True)
    lista_cluster = list(cluster['cluster'])[0:3]
    df_plot = df_passe[df_passe['cluster'].isin(lista_cluster)].reset_index(drop=True)
    # df_plot
    cor_fundo = '#2c2b2b'
    fig, ax = plt.subplots(figsize=(15,10))
    pitch = Pitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,
                    stripe=False, line_zorder=1)
    pitch.draw(ax=ax)
    def plot_scatter_df(df,cor,zo):
      pitch.scatter(df.endX, df.endY, s=200, edgecolors=cor,lw=2, c=cor_fundo, zorder=zo+1, ax=ax)
      # plt.scatter(data=df, x='to_x',y='to_y',color=cor,zorder=zo+1,label='df',edgecolors='white',s=200)
      for linha in range(len(df)):
        x_inicial = df['x'][linha]
        y_inicial = df['y'][linha]
        x_final = df['endX'][linha]
        y_final = df['endY'][linha]
        lc1 = pitch.lines(x_inicial, y_inicial,
                      x_final, y_final,
                      lw=5, transparent=True, comet=True,
                      color=cor, ax=ax,zorder=zo)
    
    lista_cor = ['#FF4E63','#8D9713','#00A6FF']
    for clus,cor in zip(lista_cluster,lista_cor):
      df = (df_plot[df_plot['cluster'] == clus].reset_index())
      df['cor'] = cor 
      plot_scatter_df(df,cor,2)
    plt.show()
    plt.savefig(f'content/cluster_{team}.png',dpi=300,facecolor=cor_fundo)
    im = Image.open(f'content/cluster_{team}.png')
    cor_fundo = '#2c2b2b'
    tamanho_arte = (3000, 2740)
    arte = Image.new('RGB',tamanho_arte,cor_fundo)
    W,H = arte.size
    im = im.rotate(90,expand=5)
    w,h= im.size
    im = im.resize((int(w/2),int(h/2)))
    im = im.copy()
    w,h= im.size
    arte.paste(im,(100,400))

    font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
    msg = f'Cluster de Passes'
    draw = ImageDraw.Draw(arte)
    w, h = draw.textsize(msg,spacing=20,font=font)
    draw.text((330,100),msg, fill='white',spacing= 20,font=font)

    font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
    msg = f'{home_team}- {away_team}'
    draw = ImageDraw.Draw(arte)
    w, h = draw.textsize(msg,spacing=20,font=font)
    draw.text((330,300),msg, fill='white',spacing= 20,font=font)

    font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
    msg = f'{team}'
    draw = ImageDraw.Draw(arte)
    w, h = draw.textsize(msg,spacing=20,font=font)
    draw.text((330,500),msg, fill='white',spacing= 20,font=font)

    fot =Image.open('Logos/Copy of pro_branco.png')
    w,h = fot.size
    fot = fot.resize((int(w/1.5),int(h/1.5)))
    fot = fot.copy()
    arte.paste(fot,(1870,1880),fot)

    times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
    logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
    try:
      r = requests.get(logo_url)
      im_bt = r.content
      image_file = io.BytesIO(im_bt)
      im = Image.open(image_file)
      w,h = im.size
      im = im.resize((int(w*2.5),int(h*2.5)))
      im = im.copy()
      arte.paste(im,(2500,100),im)
    except:
      r = requests.get(logo_url)
      im_bt = r.content
      image_file = io.BytesIO(im_bt)
      im = Image.open(image_file)
      w,h = im.size
      im = im.resize((int(w*2.5),int(h*2.5)))
      im = im.copy()
      arte.paste(im,(2500,100))

    arte.save(f'content/quadro_{grafico}_{team}.png',quality=95,facecolor='#2C2B2B')
    st.image(f'content/quadro_{grafico}_{team}.png')
    st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{team}.png', 'Imagem'), unsafe_allow_html=True)
if grafico == 'Sonar Inverso de chutes':
   def sonarinverso(df):
     shots=df[df['events']=='Shot'].reset_index(drop=True)
     def sonarplotter(dataframe):
         df = dataframe[['x','y']].reset_index(drop=True)
         # df['Y'] = df['Y']*68
         # df['X'] = df['X']*105
         df['distance'] = np.sqrt((df['x']-105)**2+(df['y']-34)**2)
         df['angle'] = np.degrees(np.arctan2(105-df['x'],34-df['y'])) + 180

         angs = [180,200,220,240,260,280,300,320,340]
         rads = []
         density = []

         for angle in angs:
             angdf = df[(df.angle > angle)&(df.angle<=angle+20)]
             median_dist = angdf.distance.median()
             rads.append(median_dist)
             density.append(len(angdf))
         md = min(density)
         Md = max(density)
         density = [(i - md)/(Md - md) for i in density]

         return (angs, rads, density, df)

     cor_fundo = '#2c2b2b'
     fig, ax = plt.subplots(figsize=(15,10))
     pitch = VerticalPitch(pitch_type='uefa', figsize=(15,10),pitch_color=cor_fundo,half=True,
                     stripe=False, line_zorder=1)
     pitch.draw(ax=ax)

     from matplotlib.colors import ListedColormap, LinearSegmentedColormap

     cmaplist = [cor_fundo, '#F43B87']
     cmap = LinearSegmentedColormap.from_list("", cmaplist)



     angs, rads, cols, sdf = sonarplotter(shots)

     for j in range(9):
         wedge = mpatches.Wedge((34, 105), rads[j], angs[j], angs[j]+20, color = cmap(cols[j]),
                                   ec = '#f7e9ec')
         ax.add_patch(wedge)
     plt.savefig(f'content/sonar_{jogador}.png',dpi=300,facecolor=cor_fundo)
     im=Image.open(f'content/sonar_{jogador}.png')
     tamanho_arte = (3000, 2740)
     arte = Image.new('RGB',tamanho_arte,cor_fundo)
     W,H = arte.size
     w,h= im.size
     im = im.resize((int(w/1.5),int(h/1.5)))
     im = im.copy()
     arte.paste(im,(-250,700))

     font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
     msg = f'Sonar Inverso de Chutes'
     draw = ImageDraw.Draw(arte)
     w, h = draw.textsize(msg,spacing=20,font=font)
     draw.text((430,100),msg, fill='white',spacing= 20,font=font)
     
     font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
     msg = f'{home_team}- {away_team}'
     draw = ImageDraw.Draw(arte)
     w, h = draw.textsize(msg,spacing=20,font=font)
     draw.text((330,350),msg, fill='white',spacing= 20,font=font)

     font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
     msg = f'{team}'
     draw = ImageDraw.Draw(arte)
     w, h = draw.textsize(msg,spacing=20,font=font)
     draw.text((430,500),msg, fill='white',spacing= 20,font=font)

     ontarget=shots[~(shots['type_displayName']=='MissedShots')].reset_index(drop=True)
     target=len(ontarget)
     total = len(shots)
     gols=shots[shots['type_displayName']=='Goal'].reset_index(drop=True)
     if gols.empty == True:
       gols=0

     font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
     msg = f'Chutes no alvo: {target} / {total}   |   Gols:  {gols} '
     draw = ImageDraw.Draw(arte)
     w, h = draw.textsize(msg,spacing=20,font=font)
     draw.text((430,650),msg, fill='white',spacing= 20,font=font)

     fot =Image.open('Logos/Copy of pro_branco.png')
     w,h = fot.size
     fot = fot.resize((int(w/2),int(h/2)))
     fot = fot.copy()
     arte.paste(fot,(2350,2200),fot)

     times_csv=pd.read_csv('csvs/_times-id (whoscored) - times-id - _times-id (whoscored) - times-id.csv')
     logo_url = times_csv[times_csv['Time'] == team].reset_index(drop=True)['Logo'][0]
     try:
       r = requests.get(logo_url)
       im_bt = r.content
       image_file = io.BytesIO(im_bt)
       im = Image.open(image_file)
       w,h = im.size
       im = im.resize((int(w*2.5),int(h*2.5)))
       im = im.copy()
       arte.paste(im,(2500,100),im)
     except:
       r = requests.get(logo_url)
       im_bt = r.content
       image_file = io.BytesIO(im_bt)
       im = Image.open(image_file)
       w,h = im.size
       im = im.resize((int(w*2.5),int(h*2.5)))
       im = im.copy()
       arte.paste(im,(2500,100))

     font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
     msg = f'*Penâltis, cobranças de falta e gol contra não incluídos'
     draw = ImageDraw.Draw(arte)
     draw.text((430,2640),msg, fill='white',spacing= 30,font=font)
     arte.save(f'content/quadro_{grafico}_{team}.png',quality=95,facecolor='#2C2B2B')
     st.image(f'content/quadro_{grafico}_{team}.png')
     st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{team}.png', 'Imagem'), unsafe_allow_html=True)
   sonarinverso(df_jogador)  
if grafico == 'PPDA':
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

    def PPDAplotter(Df):
      df = Df.copy()
      homeppda = []
      awayppda = []
      for i in range(0,int(match.expandedMinute.max())-30):
          min1 = i
          min2 = 30+i
          hppda,appda = PPDAcalculator(df,min1,min2)
          homeppda.append(hppda)
          awayppda.append(appda)
      homeppda = np.array(homeppda)
      awayppda = np.array(awayppda)
      fig,ax = plt.subplots(figsize=(15,10))
      cor_fundo='#2c2b2b'
      fig.set_facecolor(cor_fundo)
      ax.set_facecolor(cor_fundo)
      home_color = '#00ADB9'
      away_color = '#FFA966'
      ax.plot(homeppda,home_color,lw=4,ls='-',zorder=2)
      n_lines = 10
      diff_linewidth = 1.05
      alpha_value = 0.1
      ax.set_xticklabels([''])
      ax.set_ylabel("PPDA",fontsize=20,color='#edece9')
      ax.set_xlabel("Tempo (Média móvel de 30 min)",fontsize=20,color='#edece9')
      ax.yaxis.label.set_color('#edece9')
      ax.tick_params(axis='y', colors='#edece9')
      ax.plot(awayppda,away_color,lw=4,ls='-',zorder=2)
      ax.tick_params(axis='x', colors='#222222')
      spines = ['top','right','bottom','left']
      for s in spines:
          ax.spines[s].set_color('#edece9')
      ax.set_ylim(ax.get_ylim()[::-1])
      # maxminutes = df.expandedMinute.max()    
      # plt.text(s=f"{df.hometeam.unique()[0]} PPDA : "+
      #                 str(PPDAcalculator(df,0,maxminutes)[0])+'\n'+
      #             f"{df.awayteam.unique()[0]} PPDA: "+
      #                 str(PPDAcalculator(df,0,maxminutes)[1])+'\n',x = 0.25, y = 0.97,fontweight='bold',fontsize=20,color='#edece9')
      # plt.tight_layout()
      plt.savefig(f'content/PPDA_{home_team}_{away_team}.png',dpi=300,facecolor=cor_fundo)
      im=Image.open(f'content/PPDA_{home_team}_{away_team}.png')
      tamanho_arte = (3000, 2740)
      arte = Image.new('RGB',tamanho_arte,cor_fundo)
      W,H = arte.size
      w,h= im.size
      im = im.resize((int(w/1.5),int(h/1.5)))
      im = im.copy()
      arte.paste(im,(50,700))

      font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
      msg = f'PPDA'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((430,100),msg, fill='white',spacing= 20,font=font)

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'{home_team}- {away_team}'
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((430,300),msg, fill='white',spacing= 20,font=font)

      maxminutes = df.expandedMinute.max()
      h_media = str(PPDAcalculator(df,0,maxminutes)[0])
      a_media = str(PPDAcalculator(df,0,maxminutes)[1])

      font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
      msg = f'PPDA {home_team}: {h_media} - PPDA {away_team}: {a_media} '
      draw = ImageDraw.Draw(arte)
      w, h = draw.textsize(msg,spacing=20,font=font)
      draw.text((430,500),msg, fill='white',spacing= 20,font=font)

      im = Image.open('Arquivos/legenda-linha.png')
      w,h = im.size
      im = im.resize((int(w/5),int(h/5)))
      im = im.copy()
      arte.paste(im,(1300,800))

      font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
      msg = f'Mandante'
      draw = ImageDraw.Draw(arte)
      draw.text((1500,840),msg, fill='white',spacing= 30,font=font)


      font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
      msg = f'Visitante'
      draw = ImageDraw.Draw(arte)
      draw.text((1870,840),msg, fill='white',spacing= 30,font=font)

      fot =Image.open('Logos/Copy of pro_branco.png')
      w,h = fot.size
      fot = fot.resize((int(w/2),int(h/2)))
      fot = fot.copy()
      arte.paste(fot,(2350,100),fot)

      font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
      msg = f'*Quanto menor o PPDA, maior intensidade na pressão alta'
      draw = ImageDraw.Draw(arte)
      draw.text((430,2640),msg, fill='white',spacing= 30,font=font)

      arte.save(f'content/quadro_{grafico}_{home_team}_{away_team}.png',quality=95,facecolor='#2C2B2B')
      st.image(f'content/quadro_{grafico}_{home_team}_{away_team}.png')
      st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{home_team}_{away_team}.png', 'Imagem'), unsafe_allow_html=True)
    PPDAplotter(match)
    
  if grafico == 'Posse':
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

    def possplotter(Df):
        df = Df.copy()
        possession = []
        fieldtilt = []
        cor_fundo='#2c2b2b'
        for i in range(0,int(df.expandedMinute.max()-30)):
            min1 = i
            min2 = 30+i
            possession12,ft12 = possession_calc(df,min1,min2)
            possession.append(possession12)
            fieldtilt.append(ft12)
        fieldtilt = np.array(fieldtilt)
        possession = np.array(possession)
        fig,ax = plt.subplots(2,1,figsize=(15,10))
        fig.set_facecolor(cor_fundo)
        ax[0].set_facecolor(cor_fundo)
        ax[1].set_facecolor(cor_fundo)
        home_color = '#00ADB9'
        away_color = '#FFA966'
        ax[0].plot(possession,c=home_color,lw=4)
        ax[0].plot(100-possession,c=away_color,lw=4)
        ax[0].set_xticklabels('')
        ax[0].set_ylabel("Posse de bola %",fontsize=20,color='#edece9')
        ax[0].set_xlabel("Tempo (Média móvel de 30 min)",fontsize=20,color='#edece9')
        ax[0].yaxis.label.set_color('#edece9')
        ax[0].tick_params(axis='y', colors='#edece9')
        ax[0].tick_params(axis='x', colors=cor_fundo)
        ax[1].plot(fieldtilt,c=home_color,lw=4)
        ax[1].plot(100-fieldtilt,c=away_color,lw=4)
        ax[1].set_xticklabels('')
        ax[1].set_ylabel("Domínio territorial % ",fontsize=20,color='#edece9')
        ax[1].set_xlabel("Tempo (Média móvel de 30 min)",fontsize=20,color='#edece9')
        ax[1].yaxis.label.set_color('#edece9')
        ax[1].tick_params(axis='y', colors='#edece9')
        ax[1].tick_params(axis='x', colors=cor_fundo)
        ax[0].set_ylim(0,100)
        ax[1].set_ylim(0,100)

  

        spines = ['top','right','bottom','left']
        for s in spines:
            ax[0].spines[s].set_color('#edece9')
            ax[1].spines[s].set_color('#edece9')
        plt.tight_layout()
        plt.savefig(f'content/Posse_{home_team}_{away_team}.png',dpi=300,facecolor=cor_fundo)
        im=Image.open(f'content/Posse_{home_team}_{away_team}.png')
        cor_fundo = '#2c2b2b'
        tamanho_arte = (3000, 2740)
        arte = Image.new('RGB',tamanho_arte,cor_fundo)
        W,H = arte.size
        w,h= im.size
        im = im.resize((int(w/1.75),int(h/1.75)))
        im = im.copy()
        arte.paste(im,(200,900))

        font = ImageFont.truetype('Camber/Camber-Bd.ttf',150)
        msg = f'Posse de Bola'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,100),msg, fill='white',spacing= 20,font=font)

        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{home_team} - {away_team}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,300),msg, fill='white',spacing= 20,font=font)

        maxminutes = df.expandedMinute.max()
        hp_media = str(possession_calc(df,0,maxminutes)[0])
        ap_media = str(100-possession_calc(df,0,maxminutes)[0])
        hft_media = str(possession_calc(df,0,maxminutes)[1])
        aft_media = str(100-possession_calc(df,0,maxminutes)[1])
        str(possession_calc(df,0,maxminutes)[0])

        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{home_team} -> Posse : {hp_media} | Domínio territorial: {hft_media}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,450),msg, fill='white',spacing= 20,font=font)

        font = ImageFont.truetype('Camber/Camber-Rg.ttf',60)
        msg = f'{away_team} -> Posse : {ap_media} | Domínio territorial: {aft_media}'
        draw = ImageDraw.Draw(arte)
        w, h = draw.textsize(msg,spacing=20,font=font)
        draw.text((330,620),msg, fill='white',spacing= 20,font=font)

        im = Image.open('Arquivos/legenda-linha.png')
        w,h = im.size
        im = im.resize((int(w/5),int(h/5)))
        im = im.copy()
        arte.paste(im,(1200,800))

        font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
        msg = f'Mandante'
        draw = ImageDraw.Draw(arte)
        draw.text((1400,840),msg, fill='white',spacing= 30,font=font)


        font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
        msg = f'Visitante'
        draw = ImageDraw.Draw(arte)
        draw.text((1770,840),msg, fill='white',spacing= 30,font=font)

        font = ImageFont.truetype('Camber/Camber-RgItalic.ttf',40)
        msg = f'*Domínio territorial: Divisão % dos passes trocados no último terço pelas duas equipes'
        draw = ImageDraw.Draw(arte)
        draw.text((330,2640),msg, fill='white',spacing= 30,font=font)

        fot =Image.open('Logos/Copy of pro_branco.png')
        w,h = fot.size
        fot = fot.resize((int(w/2),int(h/2)))
        fot = fot.copy()
        arte.paste(fot,(2350,100),fot)
        arte.save(f'content/quadro_{grafico}_{home_team}_{away_team}.png',quality=95,facecolor='#2C2B2B')
        st.image(f'content/quadro_{grafico}_{home_team}_{away_team}.png')
        st.markdown(get_binary_file_downloader_html(f'content/quadro_{grafico}_{home_team}_{away_team}.png', 'Imagem'), unsafe_allow_html=True)
    possplotter(match)
