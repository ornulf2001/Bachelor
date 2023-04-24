#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

from funksjoner import *
#%% ---Laster inn strømfil---
fil_2 = 'Timesforbruk hele 2021 og 2022 V2.csv'
fil_3 = 'Timesforbruk hele 2021 og 2022.csv'
fil_4 = 'Timesforbruk 2022 jan-nov og desember 2021.csv'
df = pd.read_csv(fil_4, sep=';')
#%%
#---Lager egne lister for verdiene i strømforbruket---
df.iloc[0]['Time']
dato = []
time = []
forbruk = []
dato_per_dag = []
forbruk_per_dag = []
max_per_dag = []

for i in range(0,len(df)):
    if df.iloc[i]['Time'] == 'Totalt':
        dato_per_dag.append(df.iloc[i]['Dato'])
        forbruk_per_dag.append(df.iloc[i]['Forbruk'])
        max_per_dag.append(df.iloc[i]['Max'])
    else:
        dato.append(df.iloc[i]['Dato'])
        time.append(df.iloc[i]['Time'])
        forbruk.append(df.iloc[i]['Forbruk'])

#---Snur listene siden de var motsatt kronologisk i csv-fil---
forbruk.reverse()
time.reverse()
dato.reverse()
døgnfordelt = døgnfordeling(forbruk,int(len(forbruk)/24))
mean_månedforbruk = månedverdi(forbruk)
#%%
#---Plotter info om strømforbruket---
print(f'Totalt strømforbruk: {sum(forbruk)}')
måneder = ['jan', 'feb', 'mar', 'apr', 'mai', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'des']

print(mean_månedforbruk)
plt.plot(måneder,mean_månedforbruk)
plt.xlabel('Måned')
plt.ylabel('kWh/dag')
plt.show()

plt.plot(døgnfordelt)
plt.xlabel('Klokkeslett')
plt.ylabel('kWh/h')
plt.show()

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
#--- Sol ---
A = 1 # Areal sol
Zs = 0 # Retning i forhold til SØR. Varierer per tak!!!
beta = 20 # Helling på tak. Varierer

# Regner ut solproduksjon
tak_1 = solprod(Gb_n, Gd_h, Ta, antal = 1, Zs = 0, beta = 20)
tak_2 = solprod(Gb_n, Gd_h, Ta, antal = 1, Zs = 0, beta = 20)
print(sum(tak_1))
print(max(tak_1))
solproduksjon=[]
for i in range(0,8760):
    solproduksjon.append(tak_1[i]+tak_2[i])

# ---plotting av graf
# plt.plot(tak_1[0:8760])
# plt.show()

#%%
# ---Vind---

# sjekker produksjon fra en middels vindturbin
vind = vindprod(vindspeed,Ta,RH,SP, cut_in=2.8,cut_out=60,A = 12.6,Cp=0.385,n_gen=0.9)

print(sum(vind))
print(np.mean(vindspeed))
plt.plot(vind)
plt.show()
# %%
# --- Hvor mye blåser det? ---
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
plt.ylabel('Timer')
plt.grid()
plt.show()
#%%
#---Flisfyring---
bioandel = 0.2   # % av strømforbruket som kan dekkes av bio
n_bio = 0.8      # virkningsgrad bioanlegg
V_flis = 750     # kWh/lm^3, energiinnhold bio per løskubikmeter

levert_energi = [verdi*bioandel for verdi in forbruk]
flis_energi = [verdi/n_bio for verdi in levert_energi]
Vol_flis = [verdi/V_flis for verdi in flis_energi]
print(f'Energi fra bio: {round(sum(levert_energi))} kWh/år\nMaks effekt bio: {round(max(levert_energi))}')
print(f'Energi fra flis: {round(sum(flis_energi))} kWh/år\nMaks effekt flis: {round(max(flis_energi))}')
print(f'Mengde flis: {round(sum(Vol_flis))} lm^3/år\nMaks effekt kg_flis: {round(max(Vol_flis),2)}')

#%%
#---Energibalanse---
energibalanse = []
kjøpt_strøm = []
solgt_strøm = []
for i in range(0,8760):
    energi = forbruk[i]-levert_energi[i]-solproduksjon[i]-vind[i]
    if energi >= 0:
        energibalanse.append(energi)
        kjøpt_strøm.append(energi)
        solgt_strøm.append(0)
    else:
        energibalanse.append(energi)
        kjøpt_strøm.append(0)
        solgt_strøm.append(-energi)
print(f'Forbruk: {sum(forbruk)}, bio: {sum(levert_energi)}, sol: {sum(solproduksjon)}, vind: {sum(vind)}')
print(f'Energibalanse:\nsum: {sum(energibalanse)}, maks: {max(energibalanse)}, min: {min(energibalanse)}')
print(f'Kjøpt: {sum(kjøpt_strøm)}, solgt: {sum(solgt_strøm)}')
#%%
#---Optimal tilt-vinkel for bakkeplasert og sørvendt PV-panel---
vinkel_list = []
produksjon_list = []
maks_prod = 0
beste_vinkel = 0
for i in range(15,30):
    vinkel = i
    produksjon = sum(solprod(Gb_n, Gd_h, Ta, antal = 1, Zs = 0, beta = vinkel))
    if produksjon > maks_prod:
        maks_prod = produksjon
        beste_vinkel = vinkel
    elif produksjon == maks_prod: print(f'Tie ball game at vinkel {i}')
    vinkel_list.append(vinkel)
    produksjon_list.append(produksjon)
    # print(produksjon)
    # print(maks_prod)
plt.plot(vinkel_list,produksjon_list)
plt.xlabel('Tilt-vinkel')
plt.ylabel('Produksjon i kWh/år')
plt.title('Tilt-vinkel optimalisering')
plt.show()
print(f'Beste vinkel er {beste_vinkel}')
#%%
#---Optimal sør-vinkel for bakkeplasert og tiltet PV-panel---
vinkel_list = []
produksjon_list = []
maks_prod = 0
beste_vinkel = 0
for i in [-40,-20,-10,-5,0,5,10,20,40]:
    vinkel = i
    # print(vinkel)
    produksjon = sum(solprod(Gb_n, Gd_h, Ta, antal = 1, Zs = vinkel, beta = 22))
    # print(produksjon)
    if produksjon > maks_prod:
        maks_prod = produksjon
        beste_vinkel = vinkel
    elif produksjon == maks_prod: print(f'Tie ball game at vinkel {vinkel}')
    vinkel_list.append(vinkel)
    produksjon_list.append(produksjon)
    # print(produksjon)
    # print(maks_prod)
plt.plot(vinkel_list,produksjon_list)
plt.xlabel('Beta-vinkel')
plt.ylabel('Produksjon i kWh/år')
plt.title('Beta-vinkel optimalisering')
plt.show()
print(f'Beste vinkel er {beste_vinkel}')
#%%
print(f'Optimal vinkel gir: {sum(solprod(Gb_n,Gd_h,Ta,1,-20,22))}')
print(f'Uoptimal vinkel gir: {sum(solprod(Gb_n,Gd_h,Ta,1,0,0))}')

#%%
maks_prod = 0
for beta in range(21,23):
    for Z in range(-21,-19):
        prod = sum(solprod(Gb_n,Gd_h,Ta,1,Z,beta))
        if prod > maks_prod:
            tekst = f'Maks produksjon er {prod}, ved Zs = {Z} og beta = {beta}'
        print('running')
print(tekst)
#%%


#%%
#---Beregning av kostnad---
#---Nettleie---

nettleie_kr = nettleie(forbruk)
totbruk = månedtot(forbruk)
nettleie_kr
#%%
døgnfordelt_sol = døgnfordeling(solproduksjon,365)
plt.plot(døgnfordelt_sol)
plt.plot()
#%%
#---Batteri---
# eksempelbatteri
nytt_forbruk = batteri(10)
# plt.plot(døgnfordeling(nytt_forbruk,365))
# plt.show()

#%%
print(f'Nettleie før = {sum(nettleie(forbruk))}\nNettleie etter = {sum(nettleie(nytt_forbruk))}')
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

for i in range(0,8760):
    dag = int(i/24)
    time = i-dag*24+1
    if dag < 334:
        spotpris = spotpris_22_liste.iloc[dag][str(time)]
        spotpris_kr = locale.atof(spotpris)/1000
    else:
        spotpris = spotpris_21_liste.iloc[dag][str(time)]
        spotpris_kr = locale.atof(spotpris)/1000
    # print(dag)
    strømpris = spotpris_kr + påslag
    strømpris_liste.append(strømpris)
    # print(f'Strmømpris dag {dag+1} i time {time} er {strømpris} kr/MWh')
# print(strømpris_liste[8759])
plt.plot(døgnfordeling(strømpris_liste,365))
plt.title('Gjennomsnittlig strømpris gjennom døgnet')
plt.xlabel('Time')
plt.ylabel('kr/kWh')
plt.show()
#%%
#---Kostnad strøm---

plt.plot(døgnfordeling(batteri(10),365))
plt.title('Gjennomsnittlig strømpris gjennom døgnet')
plt.xlabel('Time')
plt.ylabel('kr/kWh')
plt.show()
print(f'Strømkostnad uten batteri: {sum(strømkostnad(forbruk))}\nStrømkostnad med batteri: {sum(strømkostnad(batteri(10)))}')
#%%
batteristørrelse = []
strømkost = []
for i in range(0,10000):
    batteristørrelse.append(i)
    strømkost.append(sum(strømkostnad(batteri(i))))
plt.plot(batteristørrelse,strømkost)
plt.xlabel('Batteristørrelse [kWh]')
plt.ylabel('Kostnad strøm [kr]')
plt.show()
print(len(batteri(1)))
#%%
ukesfordeling(batteri(100),dato,time)