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
df.iloc[0]['Time']
dato = []
time_liste = []
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
        time_liste.append(df.iloc[i]['Time'])
        forbruk.append(df.iloc[i]['Forbruk'])

#---Snur listene siden de var motsatt kronologisk i csv-fil---
forbruk.reverse()
time_liste.reverse()
dato.reverse()
døgnfordelt = døgnfordeling(forbruk)
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
alfa = solprod_eksperimentell(Gb_n, Gd_h, Ta, antal = 1, Zs = 29, beta = 20)
plt.plot(alfa[:])
plt.show()
#%%
#--- Sol ---
A = 1 # Areal sol
Zs = 0 # Retning i forhold til SØR. Varierer per tak!!!
beta = 20 # Helling på tak. Varierer

# Regner ut solproduksjon
sol_sanitær = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 29, beta = 20)
sol_nedre_restaurant = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = -60, beta = 15)
sol_øvre_restaurant = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = -60, beta = 35)
sol_fastmontert = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 0, beta = 22)
sol_roterende = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 666, beta = 666)
print(len(sol_sanitær))
total_solproduksjon=[]
solanlegg = [sol_sanitær, sol_nedre_restaurant, sol_øvre_restaurant, sol_roterende]
for i in range(0,8760):
    sol_prod_time = 0
    for anlegg in solanlegg:
        sol_prod_time += anlegg[i]
    total_solproduksjon.append(sol_prod_time)
print(sum(total_solproduksjon))
print(sum(sol_sanitær)+sum(sol_nedre_restaurant)+sum(sol_øvre_restaurant)+sum(sol_roterende))

# ---plotting av graf
# plt.plot(døgnfordeling(total_solproduksjon))
# plt.show()

plt.plot(døgnfordeling(sol_sanitær), label = 'Sanitærbygg')
plt.plot(døgnfordeling(sol_nedre_restaurant), label = 'Nedre tak rets.')
plt.plot(døgnfordeling(sol_øvre_restaurant), label = 'Øvre tak rets.')
# plt.plot(døgnfordeling(sol_roterende), label = 'Roterende')
# plt.plot(døgnfordeling(sol_fastmontert), label = 'Fastmontert')
# plt.plot(døgnfordeling(Gb_n/1000), label = 'Beam')
# plt.plot(døgnfordeling(Gd_h/1000), label = 'Diffuse')
plt.xlabel('Time')
plt.ylabel('kWh/h')
plt.legend()
print(f'Sum:'
      f'\n\tSanitær:     {sum(sol_sanitær)}'
      f'\n\tNedre tak:   {sum(sol_nedre_restaurant)}'
      f'\n\tØvre tak:    {sum(sol_øvre_restaurant)}'
      f'\n\tFast:        {sum(sol_fastmontert)}'
      f'\n\tRoterende:   {sum(sol_roterende)}')
#%%
#---Sjekk av sol---
def solprod_test2(Gb_n, Gd_h, Ta, antal, Zs, beta):
    '''Solproduksjon ved optimal vinkling. Tar inn fil med soldata, areal, og vinkler. Bruker dette
    til å regne ut produksjonen fra solenergi. Gitt som kWh/h'''
    # faste verdier
    L = 60.61318
    LL = 12.01088
    SL = 15
    n_sol = 0.205 # Virkningsgrad sol !!!
    LST = 0
    A = 1   # Areal per panel !!! =1.953m^2
    # Tap pga. varme
    T_tap_Pmpp = -0.0034 #Varierer per paneltype, Temperaturkoefisient Pmpp
    T_noct = 45          #Varierer per paneltype, Celletemp test
    T_a_noct = 20        # NOCT omgivelsestemp
    Gt_noct = 800
    ta = 0.9             # ta er 0.9 for silicon solar cell


    test_list = []

    for i in range(0,8760):
            LST += 1
            if LST == 25: LST = 1
            N = 1 + int(i/24)
            delta = 23.45 * sind(360/365*(284+N))
            B = (N-81)*360/364
            ET = 9.87*sind(2*B)-7.53*cosd(B)-1.5*sind(B)
            AST = LST + ET/60 - 4/60*(SL-LL) # (-<_>-) når østlig halvkule
            h = (AST - 12) * 15
            alfa = asind(sind(L)*sind(delta)+cosd(L)*cosd(delta)*cosd(h))
            z = asind(cosd(delta)*sind(h)/cosd(alfa))
            if Zs == 666 or beta == 666: #sjekker om det er roterende panelstativ. Da optimaliseres vinkelen.
                theta = 0
                beta = 90 - alfa
            else:
                theta = acosd(sind(L)*sind(delta)*cosd(beta)-cosd(L)*sind(delta)*sind(beta)*cosd(Zs)
                        +cosd(L)*cosd(delta)*cosd(h)*cosd(beta)+sind(L)*cosd(delta)*sind(beta)*cosd(h)*cosd(Zs)
                        +cosd(delta)*sind(h)*sind(beta)*sind(Zs))
            if N < 90 or N > 333: albedo = 0.65
            else: albedo = 0.2

            G = Gb_n[i] * cosd(theta) + Gd_h[i] * (180 - beta)/180 + albedo * (Gb_n[i]+Gd_h[i])*((1-cosd(theta))/2)
            if G < 0: G = 0
            P = G * n_sol

            if G != 0:
                Tc = (T_noct-T_a_noct)*(G/Gt_noct)*(1-n_sol/ta)+Ta[i]
                tap_varme = T_tap_Pmpp*(Tc-T_noct)
            else: tap_varme = 0

            produksjon = (P + tap_varme) * A * antal / 1000
            
            # if i < 48:
            #     print(f'N = {N} for dato {df.iloc[i][0]}, LST = {LST}, delta = {delta}, B = {B}, ET = {ET}, AST = {AST}, h = {h}, alfa = {alfa}')
            test_list.append(produksjon)
            # print(f'LST: {round(LST,1)}, N: {round(N,1)}, delta: {round(delta,1)}, B: {round(B,1)}, ET: {round(ET,1)}, AST: {round(AST,1)}, h: {round(h,1)}, alfa: {round(alfa,1)}, theta: {round(theta,1)} beta: {round(beta,1)}, G: {round(G,1)}')
    return test_list

sol_1 = solprod_test2(Gb_n, Gd_h, Ta, 1, 0, 0)
sol_2 = solprod_test2(Gb_n, Gd_h, Ta, 1, 666, 666)
print(sum(sol_1))
print(sum(sol_2))

#%%
# ---Vind---

# sjekker produksjon fra ulike vindturbiner
vind_vertikal = vindprod(vindspeed,Ta,RH,SP, cut_in=4,cut_out=50,A = 0.64*0.79,Cp=0.2,n_gen=0.9)
vind_vertikal2 = vindprod(vindspeed,Ta,RH,SP, cut_in=3,cut_out=50,A = 1*1,Cp=0.2,n_gen=0.9)
vind_horisontal = vindprod(vindspeed,Ta,RH,SP, cut_in= 3.1, cut_out= 49.2, A = 1.07, Cp = 0.385, n_gen = 0.9)
vind_horisontal2 = vindprod(vindspeed,Ta,RH,SP, cut_in= 3, cut_out= 50, A = np.pi*2.35**2/4, Cp = 0.385, n_gen = 0.9)

print(sum(vind_vertikal))
print(sum(vind_vertikal2))
print(sum(vind_horisontal))
print(sum(vind_horisontal2))

plt.plot(månedtot(vind_vertikal), label = 'Vertikal')
plt.plot(månedtot(vind_vertikal2), label = 'Vertikal2')
plt.plot(månedtot(vind_horisontal), label = 'Horisontal')
plt.plot(månedtot(vind_horisontal2), label = 'Horisontal2')
plt.xlabel('Time')
plt.ylabel('kWh/mnd')
plt.legend()
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
plt.ylabel('Timer [h]')
plt.grid()
plt.show()
#%%
#---Flisfyring---
bioandel = 0.21   # % av strømforbruket som kan dekkes av bio
n_bio = 0.8      # virkningsgrad bioanlegg
V_flis = 750     # kWh/lm^3, energiinnhold bio per løskubikmeter

levert_energi = [verdi*bioandel for verdi in forbruk]
flis_energi = [verdi/n_bio for verdi in levert_energi]
Vol_flis = [verdi/V_flis for verdi in flis_energi]
print(f'Energi fra bio: {round(sum(levert_energi))} kWh/år\nMaks effekt bio: {round(max(levert_energi))}')
print(f'Energi fra flis: {round(sum(flis_energi))} kWh/år\nMaks effekt flis: {round(max(flis_energi))}')
print(f'Mengde flis: {round(sum(Vol_flis))} lm^3/år\nMaks effekt kg_flis: {round(max(Vol_flis),2)}')
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
# plt.plot(døgnfordeling(strømpris_liste,365))

# plt.title('Gjennomsnittlig strømpris gjennom døgnet')
# plt.xlabel('Time')
# plt.ylabel('kr/kWh')
# plt.show()
#%%
#---Energibalanse---
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
testliste = pd.DataFrame({'Energy':energibalanse, 'Price':strømpris_liste})
# testliste.replace('.',',')
testliste.to_csv('testfil.csv', sep=';', encoding='utf-8', index=False,decimal = ',')
#%%
#---Optimal tilt-vinkel for bakkeplasert og sørvendt PV-panel---
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
døgnfordelt_sol = døgnfordeling(total_solproduksjon)
plt.plot(døgnfordelt_sol)
plt.plot()
#%%
#---Batteri---
# eksempelbatteri
print(sum(batteri(10,forbruk,time_liste)))
print(sum(forbruk))
# plt.plot(døgnfordeling(nytt_forbruk,365))
# plt.show()

#%%
print(f'Nettleie før = {sum(nettleie(forbruk))}\nNettleie etter = {sum(nettleie(nytt_forbruk))}')

#%%
#---Kostnad strøm---

plt.plot(døgnfordeling(batteri(10)))
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
print(sum(forbruk))
print(sum(batteri(100,forbruk,time_liste)))
#%%
batteristørrelse = 100
plt.plot(døgnfordeling(forbruk), label = 'Uten batteri')
plt.plot(døgnfordeling(batteri(batteristørrelse,forbruk,time_liste)), label = 'Med batteri')
plt.xlabel('Time')
plt.ylabel('kWh/h')
plt.legend()

plt.show()

plt.plot(døgnfordeling(strømkostnad(forbruk,strømpris_liste)), label = 'Uten batteri')
plt.plot(døgnfordeling(strømkostnad(batteri(batteristørrelse,forbruk,time_liste),strømpris_liste)), label = 'Med batteri')
plt.xlabel('Time')
plt.ylabel('kr/h')
plt.legend()

plt.show()
print(f'Strømkostnad\n\tUten batteri: {round(sum(strømkostnad(forbruk,strømpris_liste)),1)}\n\tMed batteri:  {round(sum(strømkostnad(batteri(batteristørrelse,forbruk,time_liste),strømpris_liste)),1)}')