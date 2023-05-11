#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
# import csv
from funksjoner import *

#%% ---Laster inn strømfil---
fil_2 = 'Timesforbruk hele 2021 og 2022 V2.csv'
fil_3 = 'Timesforbruk hele 2021 og 2022.csv'
fil_4 = 'Timesforbruk 2022 jan-nov og desember 2021.csv'
df = pd.read_csv(fil_4, sep=';')
#%%
#---Lager egne lister for verdiene i strømforbruket---
dato = []
time_liste = []
forbruk = []
# dato_per_dag = []
# forbruk_per_dag = []
# max_per_dag = []

for i in range(0,len(df)):
    if df.iloc[i]['Time'] != 'Totalt':
        # dato_per_dag.append(df.iloc[i]['Dato'])
        # forbruk_per_dag.append(df.iloc[i]['Forbruk'])
        # max_per_dag.append(df.iloc[i]['Max'])
    # else:
        dato.append(df.iloc[i]['Dato'])
        time_liste.append(df.iloc[i]['Time'])
        forbruk.append(df.iloc[i]['Forbruk'])

#---Snur listene siden de var motsatt kronologisk i csv-fil---
forbruk.reverse()
time_liste.reverse()
dato.reverse()
#%%
#---Plotter info om strømforbruket---
print(f'Totalt strømforbruk: {sum(forbruk)}')
måneder = ['jan', 'feb', 'mar', 'apr', 'mai', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'des']

print(månedverdi(forbruk))
plt.plot(måneder,månedverdi(forbruk))
plt.xlabel('Måned')
plt.ylabel('Strømforbruk [kWh/dag]')
plt.show()

plt.plot(døgnfordeling(forbruk))
plt.xlabel('Klokkeslett')
plt.ylabel('Strømforbruk [kWh]')
plt.show()

#%%
# --- TMY ---

# Laster inn fil
fil_tmy = 'tmy-data_modifisert.csv'
df = pd.read_csv(fil_tmy, sep=',')

# Egen liste for hver kolonne
G_h = df.iloc[:]['G(h)']        # Global horizontal irradiance
Gb_n = df.iloc[:]['Gb(n)']      # Direct (beam) irradiance
Gd_h = df.iloc[:]['Gd(h)']      # Diffuse horizontal irradiance
Ta = df.iloc[:]['T2m']          # Dry bulb (air) temperature
vindspeed = df.iloc[:]['WS10m'] # Windspeed
RH = df.iloc[:]['RH']           # Relative humidity %
SP = df.iloc[:]['SP']           # Surface (air) pressure
#%%
# --- Plotter alfavinkelen på lokasjonen ---
alfa = solprod_eksperimentell(Gb_n, Gd_h, Ta, antal = 1, Zs = 29, beta = 20, hvilken_verdi='alfa')
plt.plot(alfa[:])
plt.show()
#%%
#--- Sol ---

# Regner ut solproduksjon
sol_sanitær = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 29, beta = 25)
sol_nedre_restaurant = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = -60, beta = 15)
sol_øvre_restaurant = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = -60, beta = 35)
sol_fastmontert = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 0, beta = 22)
sol_roterende = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 666, beta = 666)

total_solproduksjon=[]
solanlegg = [sol_sanitær, sol_nedre_restaurant, sol_øvre_restaurant, sol_roterende]
for i in range(0,8760):
    sol_prod_time = 0
    for anlegg in solanlegg:
        sol_prod_time += anlegg[i]
    total_solproduksjon.append(sol_prod_time)

# --- Plotting av graf for gjennomsnittlig solproduksjon gjennom døgnet ---

plt.plot(døgnfordeling(sol_sanitær), label = 'Sanitærbygg')
plt.plot(døgnfordeling(sol_nedre_restaurant), label = 'Nedre tak rest.')
plt.plot(døgnfordeling(sol_øvre_restaurant), label = 'Øvre tak rest.')
plt.xlabel('Time')
plt.ylabel('Produksjon [kWh]')
plt.legend()
plt.show()

plt.plot(døgnfordeling(sol_roterende), label = 'Roterende')
plt.plot(døgnfordeling(sol_fastmontert), label = 'Fastmontert')
plt.xlabel('Time')
plt.ylabel('Produksjon [kWh]')
plt.legend()
plt.show()

plt.plot(døgnfordeling(Gb_n/1000), label = 'Beam')
plt.plot(døgnfordeling(Gd_h/1000), label = 'Diffuse')
plt.xlabel('Time')
plt.ylabel('Innstråling [kWh]')
plt.legend()
plt.show()

print(f'Årlig produksjon fra 1 panel [kWh]:'
      f'\n\tSanitær:     {round(sum(sol_sanitær),2)}'
      f'\n\tNedre tak:   {round(sum(sol_nedre_restaurant),2)}'
      f'\n\tØvre tak:    {round(sum(sol_øvre_restaurant),2)}'
      f'\n\tFast:        {round(sum(sol_fastmontert),2)}'
      f'\n\tRoterende:   {round(sum(sol_roterende),2)}')
#%%
# ---Vind---

# sjekker produksjon fra ulike vindturbiner
vind_vertikal = vindprod(vindspeed,Ta,RH,SP, cut_in=4,cut_out=50,A = 0.64*0.79,Cp=0.2,n_gen=0.9)
vind_vertikal2 = vindprod(vindspeed,Ta,RH,SP, cut_in=3,cut_out=50,A = 1*1,Cp=0.2,n_gen=0.9)
vind_horisontal = vindprod(vindspeed,Ta,RH,SP, cut_in= 3.1, cut_out= 49.2, A = 1.07, Cp = 0.385, n_gen = 0.9)
vind_horisontal2 = vindprod(vindspeed,Ta,RH,SP, cut_in= 3, cut_out= 50, A = np.pi*2.35**2/4, Cp = 0.385, n_gen = 0.9)
vind = []
for i in range(0,8760):
    vind.append(vind_vertikal[i]+vind_vertikal2[i]+vind_horisontal[i]+vind_horisontal2[i])

print(f'Årlig produksjon fra vindturbiner [kWh]:'
      f'\n\tAtlas X7 (V):   {round(sum(vind_vertikal),2)}'
      f'\n\tAtlas 7 (V):    {round(sum(vind_vertikal2),2)}'
      f'\n\tMaster 1kW (H): {round(sum(vind_horisontal),2)}'
      f'\n\tMagnum 5 (H):   {round(sum(vind_horisontal2),2)}')

plt.plot(måneder,månedtot(vind_vertikal), label = 'Atlas X7 (V)')
plt.plot(månedtot(vind_vertikal2), label = 'Atlas 7 (V)')
plt.plot(månedtot(vind_horisontal), label = 'Master 1kW (H)')
plt.plot(månedtot(vind_horisontal2), label = 'Magnum 5 (H)')
plt.xlabel('Måned')
plt.ylabel('Produksjon [kWh]')
plt.legend()
plt.show()
# %%
# --- Plot av fordelingen av ulike vindhastigheter ---
vindhastighet = ['0-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10','10<']
vindtimer = [0,0,0,0,0,0,0,0,0,0,0]
for v in vindspeed:
    if v>10: vindtimer[10]+=1
    elif v>9: vindtimer[9]+=1
    elif v>8: vindtimer[8]+=1
    elif v>7: vindtimer[7]+=1
    elif v>6: vindtimer[6]+=1
    elif v>5: vindtimer[5]+=1
    elif v>4: vindtimer[4]+=1
    elif v>3: vindtimer[3]+=1
    elif v>2: vindtimer[2]+=1
    elif v>1: vindtimer[1]+=1
    else: vindtimer[0] +=1
print(vindhastighet)
print(vindtimer)
plt.plot(vindhastighet,vindtimer)
plt.xlabel('Vindhastighet [m/s]')
plt.ylabel('Timer [h]')
plt.grid()
plt.show()
#%%
#---Flisfyring---
bioandel = 0.21   # % av strømforbruket som kan dekkes av bio
n_bio = 0.8       # virkningsgrad bioanlegg
V_flis = 750      # kWh/lm^3, energiinnhold bio per løskubikmeter

levert_energi = [verdi*bioandel for verdi in forbruk]
flis_energi = [verdi/n_bio for verdi in levert_energi]
Vol_flis = [verdi/V_flis for verdi in flis_energi]
print(f'Energi fra bio: {round(sum(levert_energi))} kWh/år\nMaks effekt bio: {round(max(levert_energi))}')
print(f'Energi fra flis: {round(sum(flis_energi))} kWh/år\nMaks effekt flis: {round(max(flis_energi))}')
print(f'Mengde flis: {round(sum(Vol_flis))} lm^3/år\nMaks effekt kg_flis: {round(max(Vol_flis),2)}')
#%%
#---Strømpris---
# Lager liste for strømpris og spotpris
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

# Plot av gjennomsnittlig strømpris gjennom døgnet
plt.plot(døgnfordeling(strømpris_liste))
plt.title('Gjennomsnittlig strømpris gjennom døgnet')
plt.xlabel('Time')
plt.ylabel('Strømpris [kr/kWh]')
plt.show()
#%%
#---Energibalanse---
# Ser på energibalansen når noe av strømforbruket dekkes fra egne energikilder
energibalanse = []
kjøpt_strøm = []
solgt_strøm = []
for i in range(0,8760):
    energi = forbruk[i]-levert_energi[i]-total_solproduksjon[i]-vind[i]
    if energi >= 0:
        energibalanse.append(energi)
        kjøpt_strøm.append(energi)
        solgt_strøm.append(0)
    else:
        energibalanse.append(energi)
        kjøpt_strøm.append(0)
        solgt_strøm.append(-energi)
print(f'Forbruk: {sum(forbruk)}, bio: {sum(levert_energi)}, sol: {sum(total_solproduksjon)}, vind: {sum(vind)}')
print(f'Energibalanse:\nsum: {sum(energibalanse)}, maks: {max(energibalanse)}, min: {min(energibalanse)}')
print(f'Kjøpt: {sum(kjøpt_strøm)}, solgt: {sum(solgt_strøm)}')
# testliste = pd.DataFrame({'Energy':energibalanse, 'Price':strømpris_liste})
# testliste.replace('.',',')
# testliste.to_csv('testfil.csv', sep=';', encoding='utf-8', index=False,decimal = ',')
#%%
#---Optimal tilt-vinkel for bakkeplasert og sørvendt PV-panel---
# OBS! Tar lang tid
vinkel_list = []
produksjon_list = []
maks_prod = 0
beste_vinkel = 0
for i in range(15,30):
    vinkel = i
    produksjon = sum(solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 0, beta = vinkel))
    if produksjon > maks_prod:
        maks_prod = produksjon
        beste_vinkel = vinkel
    elif produksjon == maks_prod: print(f'Tie ball game at vinkel {i}')
    vinkel_list.append(vinkel)
    produksjon_list.append(produksjon)
plt.plot(vinkel_list,produksjon_list)
plt.xlabel('Tilt-vinkel')
plt.ylabel('Produksjon i kWh/år')
plt.title('Tilt-vinkel optimalisering')
plt.show()
print(f'Beste vinkel er {beste_vinkel}')
#%%
#---Optimal sør-vinkel for bakkeplasert og tiltet PV-panel---
# OBS. Tar lang tid
vinkel_list = []
produksjon_list = []
maks_prod = 0
beste_vinkel = 0
for i in [-40,-20,-10,-5,0,5,10,20,40]:
    vinkel = i
    produksjon = sum(solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = vinkel, beta = 22))
    if produksjon > maks_prod:
        maks_prod = produksjon
        beste_vinkel = vinkel
    elif produksjon == maks_prod: print(f'Tie ball game at vinkel {vinkel}')
    vinkel_list.append(vinkel)
    produksjon_list.append(produksjon)
plt.plot(vinkel_list,produksjon_list)
plt.xlabel('Beta-vinkel')
plt.ylabel('Produksjon i kWh/år')
plt.title('Beta-vinkel optimalisering')
plt.show()
print(f'Beste vinkel er {beste_vinkel}')
#%%
#---Beregning av kostnad---
#---Nettleie---

nettleie_kr = nettleie(forbruk)
totbruk = månedtot(forbruk)
nettleie_kr
#%%
#---Batteri---
# eksempelbatteri
print(sum(batteri(10,forbruk,time_liste)))
print(sum(forbruk))
# plt.plot(døgnfordeling(nytt_forbruk,365))
# plt.show()
#%%
#---Kostnad strøm---

#%%
print(sum(forbruk))
print(sum(batteri(100,forbruk,time_liste)))
#%%
# Ser på påvirkningen på strømforbruket ved bruk av batterier
batteristørrelse = 100
plt.plot(døgnfordeling(forbruk), label = 'Uten batteri')
plt.plot(døgnfordeling(batteri(batteristørrelse,forbruk,time_liste)), label = 'LAB')
plt.plot(døgnfordeling(batteri_2(batteristørrelse,forbruk,time_liste,0.95)), label = 'LIB')
plt.plot(døgnfordeling(batteri_2(batteristørrelse,forbruk,time_liste,1)), label = 'Ideell')
plt.xlabel('Time')
plt.ylabel('kWh')
plt.legend()

plt.show()

plt.plot(døgnfordeling(strømkostnad(forbruk,strømpris_liste,spotpris_liste)), label = 'Uten batteri')
plt.plot(døgnfordeling(strømkostnad(batteri(batteristørrelse,forbruk,time_liste),strømpris_liste,spotpris_liste)), label = 'LAB')
plt.plot(døgnfordeling(strømkostnad(batteri_2(batteristørrelse,forbruk,time_liste,0.95),strømpris_liste,spotpris_liste)), label = 'LIB')
plt.plot(døgnfordeling(strømkostnad(batteri_2(batteristørrelse,forbruk,time_liste,1),strømpris_liste,spotpris_liste)), label = 'Ideell')
plt.xlabel('Time')
plt.ylabel('kr')
plt.legend()

plt.show()
print(f'Strømforbruk [kWh/år]\n\tUten:   {round(sum(forbruk),1)}\n\tLAB:    {round(sum(batteri(batteristørrelse,forbruk,time_liste)),1)}\n\tLIB:    {round(sum(batteri_2(batteristørrelse,forbruk,time_liste,0.95)),1)}\n\tIdeell: {round(sum(batteri_2(batteristørrelse,forbruk,time_liste,1)),1)}')
print(f'Strømkostnad [kr/år]\n\tUten:   {round(sum(strømkostnad(forbruk,strømpris_liste,spotpris_liste)),1)}\n\tLAB:    {round(sum(strømkostnad(batteri(batteristørrelse,forbruk,time_liste),strømpris_liste,spotpris_liste)),1)}\n\tLIB:    {round(sum(strømkostnad(batteri_2(batteristørrelse,forbruk,time_liste,0.95),strømpris_liste,spotpris_liste)),1)}\n\tIdeell: {round(sum(strømkostnad(batteri_2(batteristørrelse,forbruk,time_liste,1),strømpris_liste,spotpris_liste)),1)}')