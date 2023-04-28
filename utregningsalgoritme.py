#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
# import csv

from funksjoner import *
#%% ---Laster inn strømfil---
fil_4 = 'Timesforbruk 2022 jan-nov og desember 2021.csv'
df = pd.read_csv(fil_4, sep=';')
#%%
#---Lager egne lister for verdiene i strømforbruket---
df.iloc[0]['Time']
dato_liste = []
time_liste = []
energiforbruk_liste = []
dato_per_dag = []
forbruk_per_dag = []
max_per_dag = []

for i in range(0,len(df)):
    if df.iloc[i]['Time'] == 'Totalt':
        dato_per_dag.append(df.iloc[i]['Dato'])
        forbruk_per_dag.append(df.iloc[i]['Forbruk'])
        max_per_dag.append(df.iloc[i]['Max'])
    else:
        dato_liste.append(df.iloc[i]['Dato'])
        time_liste.append(df.iloc[i]['Time'])
        energiforbruk_liste.append(df.iloc[i]['Forbruk'])
#---Snur listene siden de var motsatt kronologisk i csv-fil---
energiforbruk_liste.reverse()
time_liste.reverse()
dato_liste.reverse()
#%%
# --- TMY ---

# Laster inn fil
fil_tmy = 'tmy-data_modifisert.csv'
df = pd.read_csv(fil_tmy, sep=',')

# Egen liste for hver kolonne
G_h = df.iloc[:]['G(h)']   # Global horizontal irradiance
Gb_n = df.iloc[:]['Gb(n)'] # Direct (beam) irradiance
Gd_h = df.iloc[:]['Gd(h)'] # Diffuse horizontal irradiance
Ta = df.iloc[:]['T2m']     # Dry bulb (air) temperature
vindspeed = df.iloc[:]['WS10m'] # Windspeed
RH = df.iloc[:]['RH'] # Relative humidity %
SP = df.iloc[:]['SP'] # Surface (air) pressure
#%%
# Regner ut solproduksjon
sol_sanitær = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 20, beta = 20)
sol_nedre_restaurant = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = -60, beta = 15)
sol_øvre_restaurant = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = -60, beta = 35)
sol_fastmontert = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 0, beta = 22)
sol_roterende = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 666, beta = 666)
#%%
# ---Vind---

# sjekker produksjon fra ulike vindturbiner
vind_vertikal = vindprod(vindspeed,Ta,RH,SP, cut_in=4,cut_out=50,A = 0.64*0.79,Cp=0.2,n_gen=0.9)
vind_vertikal2 = vindprod(vindspeed,Ta,RH,SP, cut_in=3,cut_out=50,A = 1*1,Cp=0.2,n_gen=0.9)
vind_horisontal = vindprod(vindspeed,Ta,RH,SP, cut_in= 3.1, cut_out= 49.2, A = 1.07, Cp = 0.385, n_gen = 0.9)
vind_horisontal2 = vindprod(vindspeed,Ta,RH,SP, cut_in= 3, cut_out= 50, A = np.pi*2.35**2/4, Cp = 0.385, n_gen = 0.9)
#%%
#---Flisfyring---
bioandel = 0.21   # % av strømforbruket som kan dekkes av bio
n_bio = 0.8      # virkningsgrad bioanlegg
V_flis = 750     # kWh/lm^3, energiinnhold bio per løskubikmeter

levert_energi = [verdi*bioandel for verdi in energiforbruk_liste]
flis_energi = [verdi/n_bio for verdi in levert_energi]
Vol_flis = [verdi/V_flis for verdi in flis_energi]
#%%
#---Strømpris---
import locale
locale.setlocale(locale.LC_ALL, '')
spotpris_22 = 'spotprisoslo22-mod.csv'
spotpris_22_liste = pd.read_csv(spotpris_22, sep=';')
spotpris_21 = 'spotprisoslo21-mod.csv'
spotpris_21_liste = pd.read_csv(spotpris_21, sep=';')
påslag = 0.049 # kr/kWh
strømpris_liste = []
spotpris_liste = []

for i in range(0,8760):
    dag = int(i/24)
    time = i-dag*24+1
    if dag < 334:
        spotpris = spotpris_22_liste.iloc[dag][str(time)]
        spotpris_kr = locale.atof(spotpris)/1000
    else:
        spotpris = spotpris_21_liste.iloc[dag][str(time)]
        spotpris_kr = locale.atof(spotpris)/1000
    strømpris = spotpris_kr + påslag
    strømpris_liste.append(strømpris)
    spotpris_liste.append(spotpris_kr)
#%%
#---Energibalanse---
energibalanse = []
for i in range(0,8760):
    energi = energiforbruk_liste[i]-levert_energi[i]-total_solproduksjon[i]-vind[i]
    energibalanse.append(energi)

#%%
#---Batteri---
energibalanse = batteri(10,energibalanse,time_liste)
#%%
#---Beregning av kostnad---
#---Nettleie---

nettleie_kr = nettleie(energibalanse)
totbruk = månedtot(energibalanse)
nettleie_kr
#---Strømkostnad---
strømkostnaden = strømkostnad(energiforbruk_liste,strømpris_liste)
#---Installasjon---
