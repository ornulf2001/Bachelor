#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
# import csv

from funksjoner import *
#---Laster inn strømfil---
fil_4 = 'Timesforbruk 2022 jan-nov og desember 2021.csv'
df = pd.read_csv(fil_4, sep=';')

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
#---Variabler---
paneler_sanitær = 1             # antall
paneler_nedre_restaurant = 1
paneler_øvre_restaurant = 1
paneler_fastmontert = 1
paneler_roterende = 1

vindturbiner_v1 = 1             # antall
vindturbiner_v2 = 1
vindturbiner_h1 = 1
vindturbiner_h2 = 1

bioandel = 0.21                 # % av strømforbruket som kan dekkes av bio
batterikapasitet = 10           # kWh lagringskapasitet

#---Priser og kostnad---
flis = 0.4 # kr/kWh
#---Komponenter---
PV_panel = 0
vindturbin_v1 = 0
vindturbin_v2 = 0
vindturbin_h1 = 0
vindturbin_h2 = 0
batteribank = 0
fliskjele = 0
#---Installasjon---
pris_sol = PV_panel*(paneler_roterende+paneler_nedre_restaurant+paneler_øvre_restaurant+paneler_fastmontert+paneler_roterende)

# Regner ut solproduksjon
sol_sanitær = solprod_2(Gb_n, Gd_h, Ta, antal = paneler_sanitær, Zs = 20, beta = 20)
sol_nedre_restaurant = solprod_2(Gb_n, Gd_h, Ta, antal = paneler_nedre_restaurant, Zs = -60, beta = 15)
sol_øvre_restaurant = solprod_2(Gb_n, Gd_h, Ta, antal = paneler_øvre_restaurant, Zs = -60, beta = 35)
sol_fastmontert = solprod_2(Gb_n, Gd_h, Ta, antal = paneler_fastmontert, Zs = 0, beta = 22)
sol_roterende = solprod_2(Gb_n, Gd_h, Ta, antal = paneler_roterende, Zs = 666, beta = 666)
total_solproduksjon=[]
solanlegg = [sol_sanitær, sol_nedre_restaurant, sol_øvre_restaurant, sol_fastmontert, sol_roterende]
for i in range(0,8760):
    sol_prod_time = 0
    for anlegg in solanlegg:
        sol_prod_time += anlegg[i]
    total_solproduksjon.append(sol_prod_time)

# ---Vind---

# sjekker produksjon fra ulike vindturbiner
vind_vertikal = vindturbiner_v1 * vindprod(vindspeed,Ta,RH,SP, cut_in=4,cut_out=50,A = 0.64*0.79,Cp=0.2,n_gen=0.9)
vind_vertikal2 = vindturbiner_v2 * vindprod(vindspeed,Ta,RH,SP, cut_in=3,cut_out=50,A = 1*1,Cp=0.2,n_gen=0.9)
vind_horisontal = vindturbiner_h1 * vindprod(vindspeed,Ta,RH,SP, cut_in= 3.1, cut_out= 49.2, A = 1.07, Cp = 0.385, n_gen = 0.9)
vind_horisontal2 = vindturbiner_h2 * vindprod(vindspeed,Ta,RH,SP, cut_in= 3, cut_out= 50, A = np.pi*2.35**2/4, Cp = 0.385, n_gen = 0.9)
total_vindproduksjon=[]
vindanlegg = [vind_vertikal, vind_vertikal2, vind_horisontal, vind_horisontal2]
for i in range(0,8760):
    vind_prod_time = 0
    for anlegg in vindanlegg:
        vind_prod_time += anlegg[i]
    total_vindproduksjon.append(vind_prod_time)

#---Flisfyring---
n_bio = 0.8      # virkningsgrad bioanlegg
V_flis = 750     # kWh/lm^3, energiinnhold bio per løskubikmeter

levert_energi = [verdi*bioandel for verdi in energiforbruk_liste]
flis_energi = [verdi/n_bio for verdi in levert_energi]
Vol_flis = [verdi/V_flis for verdi in flis_energi]


#---Energibalanse---
energibalanse = []
for i in range(0,8760):
    energi = energiforbruk_liste[i]-levert_energi[i]-total_solproduksjon[i]-total_vindproduksjon[i]
    energibalanse.append(energi)


#---Batteri---
energibalanse = batteri(batterikapasitet,energibalanse,time_liste)

#---Beregning av kostnad---
#---Nettleie---

nettleie_kr = nettleie(energibalanse)
#---Strømkostnad---
strømkostnaden = strømkostnad(energibalanse,strømpris_liste,spotpris_liste)

total_kostnad = sum(strømkostnaden)+49*12 + sum(nettleie_kr) + sum(flis_energi)*flis # + innstallasjonskostnad/levetid ? + vedlikehold
print(f'Total kostnad før {sum(strømkostnad(energiforbruk_liste,strømpris_liste,spotpris_liste)+49*12+nettleie(energiforbruk_liste))} kr/år')
print(f'Total kostnad etter {total_kostnad} kr/år')

