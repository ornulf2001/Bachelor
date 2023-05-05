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
print('Lister er lastet inn')
#%%
#              san, nedre, øvre, fast, rot, bio, batt
energikilde = [  1,     1,    1,    0,   0,   0,  0]

#---Variabler---
paneler_sanitær = 28             # antall
paneler_nedre_restaurant = 20
paneler_øvre_restaurant = 32
paneler_fastmontert = 24*energikilde[3]
paneler_roterende = 15*energikilde[4]

vindturbiner_v1 = 0             # antall
vindturbiner_v2 = 0
vindturbiner_h1 = 0
vindturbiner_h2 = 0

bioandel = 0.21*energikilde[5]          # % av strømforbruket som kan dekkes av bio
batterikapasitet = energikilde[6]       # kWh lagringskapasitet

#---Priser og kostnad---
flis_pris = 0.4   # kr/kWh
#---Komponenter---
PV_panel = 2279
festeklemme = 29
festeskinne = 595
fast_stativ = 33500
rot_stativ = 100000
inverter = 15000
sol_installasjon = 50000

PV_tak_sanitær = festeskinne*24+festeklemme*96+PV_panel*paneler_sanitær+sol_installasjon
PV_tak_nedre = festeskinne*15+festeklemme*66+PV_panel*paneler_nedre_restaurant+sol_installasjon
PV_tak_øvre = festeskinne*24+festeklemme*102+PV_panel*paneler_øvre_restaurant+sol_installasjon
PV_fri = fast_stativ+paneler_fastmontert*PV_panel+sol_installasjon
PV_rot = rot_stativ+PV_panel*paneler_roterende+sol_installasjon

# vindturbin_v1 = 16580
# vindturbin_v2 = 17620
# vindturbin_h1 = 12600
# vindturbin_h2 = 20270
# installasjon_vind = 0
batteribank = 4595/(12*260)*1000*batterikapasitet  #kr/kWh * kWh
installasjon_batteri = 10000*min(1,energikilde[6])
fliskjele = 800000
installasjon_bio = 500000
#---Installasjon---
pris_sol = PV_tak_sanitær*energikilde[0] + PV_tak_nedre*energikilde[1] + PV_tak_øvre*energikilde[2] + PV_fri*energikilde[3] + PV_rot*energikilde[4]
pris_invertere = inverter*(energikilde[0]+energikilde[1]+energikilde[2]+min(1,energikilde[3])+min(1,energikilde[4])+min(1,energikilde[6]))

installasjonskostnader = pris_sol + pris_invertere + (batteribank+installasjon_batteri) + energikilde[5]*(fliskjele+installasjon_bio)
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
vindantall = [vindturbiner_v1, vindturbiner_v2, vindturbiner_h1, vindturbiner_h2]
for i in range(0,8760):
    vind_prod_time = 0
    for anlegg in vindanlegg:
        if len(anlegg)!=0:
            vind_prod_time += anlegg[i]
    total_vindproduksjon.append(vind_prod_time)

#---Flisfyring---
n_bio = 0.8      # virkningsgrad bioanlegg
V_flis = 750     # kWh/lm^3, energiinnhold bio per løskubikmeter

levert_energi = [verdi*bioandel for verdi in energiforbruk_liste]
flis_energi = [verdi/n_bio for verdi in levert_energi]
Vol_flis = [verdi/V_flis for verdi in flis_energi]


#---Energibalanse før batteri---
energibalanse = []

for i in range(0,8760):
    energi = energiforbruk_liste[i]-levert_energi[i]-total_solproduksjon[i]-total_vindproduksjon[i]
    if energi >= 0:
        energibalanse.append(energi)
    else:
        energibalanse.append(energi)

#---Batteri---
energibalanse_batt = batteri(batterikapasitet,energibalanse,time_liste)
#---Energibalanse etter batteri---
kjøpt_strøm = []
solgt_strøm = []
for energi in energibalanse_batt:
    if energi >= 0:
        kjøpt_strøm.append(energi)
        solgt_strøm.append(0)
    else:
        kjøpt_strøm.append(0)
        solgt_strøm.append(-energi)

#---Plot---

plt.plot(døgnfordeling(energiforbruk_liste))
plt.plot(døgnfordeling(energibalanse))
plt.plot(døgnfordeling(energibalanse_batt))
plt.show()
#---Beregning av kostnad---
#---Nettleie---

nettleie_kr = nettleie(energibalanse_batt)
#---Strømkostnad---
strømkostnaden = strømkostnad(energibalanse_batt,strømpris_liste,spotpris_liste)
total_årlig_kostnad = sum(strømkostnaden)+49*12 + sum(nettleie_kr) + sum(flis_energi)*flis_pris# + installasjonskostnader# + innstallasjonskostnad/levetid ? + vedlikehold

print(f'Total årlig kostnad før {round(sum(strømkostnad(energiforbruk_liste,strømpris_liste,spotpris_liste))+49*12+sum(nettleie(energiforbruk_liste)))} kr/år ekskl. MVA')
print(f'Total årlig kostnad etter {round(total_årlig_kostnad)} kr/år ekskl. MVA')
print(f'Installasjonskostnader: {installasjonskostnader}')
print(f'Nettleie: {sum(nettleie(energiforbruk_liste))}---{sum(nettleie(energibalanse_batt))}')
print(f'Strømforbruk: {sum(energibalanse_batt)}'
      f'\nKjøpt strøm:  {sum(kjøpt_strøm)}'
      f'\nSolgt strøm:    {sum(solgt_strøm)}')
print(f'\nBio\n'
      f'\tInstallasjon/år: {energikilde[5]*(fliskjele+installasjon_bio)/30}\n'
      f'\tKostnad flis:    {sum(flis_energi)*flis_pris}\n'
      f'\tBespart:         {sum(strømkostnad(levert_energi,strømpris_liste,spotpris_liste))}')