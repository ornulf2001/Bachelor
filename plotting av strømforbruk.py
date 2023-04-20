#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def døgnverdi(liste):
    '''Lager liste med total produksjon per døgn'''
    døgnverdi = []
    for i in range(0,365):
        døgnverdi.append(sum(liste[(0+24*i):(24+24*i)]))
    return døgnverdi

def månedverdi(liste):
    '''Gjør om timesverdier til daglig gjennomsnitt for hver måned'''
    månedlengde = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    månedliste = []
    start, slutt = 0, 0
    for i in range(0,12):
        slutt += 24*månedlengde[i]
        månedliste.append(sum(liste[start:slutt])/månedlengde[i])
        # print(slutt)
        start += 24*månedlengde[i]
    return månedliste

def døgnfordeling(måned,døgn):
    '''Viser fordelingen av i løpet av timene i døgnet'''
    time = np.zeros(24)
    for t in range(0,24):
        for d in range(0,døgn):
            time[t] += måned[d*24+t]
    time = time/døgn
    return time

#%%
# --- Strømforbruk ---

# fil = 'Verdier Bacheloroppgave - Timesdata strøm.csv'
# df = pd.read_csv(fil, sep=',')
# juli = df.iloc[:]['Verdi juli']
# august = df.iloc[:]['Verdi august']
# september = df.iloc[:]['Verdi september']
# def timezwapper(måned,døgn):
#     måned_dag = np.zeros(døgn)
#     for d in range(0,døgn):
#         for t in range(0,24):
#             måned_dag[d] += måned[d*24+t]
#     return måned_dag

# juli_dag = timezwapper(juli,31)
# august_dag = timezwapper(august,31)
# september_dag = timezwapper(september,30)

# def stats(måned):
#     return str(round(np.sum(måned),2)) + '\t\t' + str(round(np.mean(måned),2)) + '\n'
# #print(stats(juli_dag))
# print('Sum per måned: \t\tAvg per dag:\n'+ 'Juli:\t\t' + stats(juli_dag) + 'August:\t\t' + stats(august_dag) + 'September:\t' + stats(september_dag))

# #print(døgnfordeling(juli,31))
# plt.plot(døgnfordeling(juli,31))
# plt.show()

#%% ---Laster inn strømfil---
fil_2 = 'Timesforbruk hele 2021 og 2022 V2.csv'
fil_3 = 'Timesforbruk hele 2021 og 2022.csv'
fil_4 = 'Timesforbruk 2022 jan-nov og desember 2021.csv'
df = pd.read_csv(fil_4, sep=';')

#%% ---Feilsøking---
# total = 0
# time = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#np.zeros(24)
# for i in range(0,len(df)):
#     if df.iloc[i]['Time'] == 'Totalt':
#         total += 1
#         if df.iloc[i+1]['Time'] == '22':
#             print(f'hey: {i}')
#     elif df.iloc[i]['Time'] == 'Time':
#         pass
#     else:
#         val = int(df.iloc[i]['Time'])
#         time[val] += 1
# print(total)
# print(time)
#%%
#data = df
#data = data.split(';')
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


forbruk.reverse()
time.reverse()
dato.reverse()
døgnfordelt = døgnfordeling(forbruk,int(len(forbruk)/24))
mean_månedforbruk = månedverdi(forbruk)
print(f'Totalt strømforbruk: {sum(forbruk)}')
måneder = ['jan', 'feb', 'mar', 'apr', 'mai', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'des']

print(mean_månedforbruk)
plt.plot(måneder,mean_månedforbruk)
plt.xlabel('Måned')
plt.ylabel('kWh/dag')
plt.show()
# time
# int(len(forbruk)/24)
# forbruk
plt.plot(døgnfordelt)
plt.xlabel('Klokkeslett')
plt.ylabel('kWh/h')
plt.show()


#%%
#---Maks effekt per måned---
def månedmaks(årsliste):
    '''Finner høyeste verdi for hver måned. Fin for å finne makseffekt'''
    måned_maks = [0,0,0,0,0,0,0,0,0,0,0,0]
    døgntot = 0
    for i,val in enumerate([31,28,31,30,31,30,31,31,30,31,30,31]):
        måned_maks[i] = max(årsliste[24*døgntot:24*(døgntot+val)])
        døgntot += val
    return måned_maks
print(månedmaks(forbruk))

#---Totalt forbruk per måned---

def månedtot(årsliste):
    '''Finner total verdi for hver måned. Fin for å finne strømforbruk'''
    måned_tot = [0,0,0,0,0,0,0,0,0,0,0,0]
    døgntot = 0
    for i,val in enumerate([31,28,31,30,31,30,31,31,30,31,30,31]):
        måned_tot[i] = round(sum(årsliste[24*døgntot:24*(døgntot+val)]),1)
        døgntot += val
    return måned_tot
print(månedtot(forbruk))

#%%
# --- Sol ---

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
A = 1 # Areal sol
Zs = 0 # Retning i forhold til SØR. Varierer per tak!!!
beta = 20 # Helling på tak. Varierer

# Definerer trigonometriske funksjoner for bruk med grader
sind = lambda degrees: np.sin(np.deg2rad(degrees))
cosd = lambda degrees: np.cos(np.deg2rad(degrees))
asind = lambda degrees: np.rad2deg(np.arcsin(degrees))
acosd = lambda degrees: np.rad2deg(np.arccos(degrees))

# This is where the magic happens! <3
def solprod(Gb_n, Gd_h, Ta, antal, Zs, beta):
    '''Tar inn fil med soldata, areal, og vinkler. Bruker dette
    til å regne ut produksjonen fra solenergi. Gitt som kWh/h'''
    # faste verdier
    L = 60.61318
    LL = 12.01088
    SL = 15
    n_sol = 0.205 # Virkningsgrad sol !!!
    LST = -1
    A = 1.722*1.134   # Areal per panel !!! =1.953m^2
    # Tap pga. varme
    T_tap_Pmpp = -0.0034 #Varierer per paneltype, Temperaturkoefisient Pmpp
    T_noct = 45          #Varierer per paneltype, Celletemp test
    T_a_noct = 20        # NOCT omgivelsestemp
    Gt_noct = 800
    ta = 0.9             # ta er 0.9 for silicon solar cell


    test_list = []

    for i,val in enumerate(Gb_n):
            LST += 1
            if LST == 24: LST = 0
            N = 1 + int(i/24)
            delta = 23.45 * sind(360/365*(284+N))
            B = (N-81)*360/364
            ET = 9.87*sind(2*B)-7.53*cosd(B)-1.5*sind(B)
            AST = LST + ET/60 - 4/60*(SL-LL) # (-<_>-) når østlig halvkule
            h = (AST - 12) * 15
            alfa = asind(sind(L)*sind(delta)+cosd(L)*cosd(delta)*cosd(h))
            z = asind(cosd(delta)*sind(h)/cosd(alfa))
            theta = acosd(sind(L)*sind(delta)*cosd(beta)-cosd(L)*sind(delta)*sind(beta)*cosd(Zs)
                        +cosd(L)*cosd(delta)*cosd(h)*cosd(beta)+sind(L)*cosd(delta)*sind(beta)*cosd(h)*cosd(Zs)
                        +cosd(delta)*sind(h)*sind(beta)*sind(Zs))
            if N < 90 or N > 333: albedo = 0.65
            else: albedo = 0.2
            
            
            G = Gb_n[i] * cosd(theta) + Gd_h[i] * (180 - beta)/180 + albedo * (Gb_n[i]+Gd_h[i])*((1-cosd(theta))/2)
            if G < 0: G = 0
            P = G * n_sol

            Tc = (T_noct-T_a_noct)*(G/Gt_noct)*(1-n_sol/ta)+Ta[i]
            tap_varme = T_tap_Pmpp*(Tc-T_noct)

            produksjon = (P + tap_varme) * A * antal / 1000
            
            # if i < 48:
            #     print(f'N = {N} for dato {df.iloc[i][0]}, LST = {LST}, delta = {delta}, B = {B}, ET = {ET}, AST = {AST}, h = {h}, alfa = {alfa}')
            test_list.append(produksjon)
    return test_list

#%%

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
def luft_tetthet(Ta,RH,SP):
    '''Bruker temperatur, luftfuktighet og trykk til å regne ut lufttettheten'''
    T = Ta + 273.15
    Rd = 287.05
    e_so = 6.1078
    c0 = 0.99999683
    c1 = -0.90826951e-2
    c2 = 0.78736169e-4
    c3 = -0.61117958e-6
    c4 = -0.43884187e-8
    c5 = -0.29883885e-10
    c6 = 0.21874425e-12
    c7 = -0.17892321e-14
    c8 = 0.11112018e-16
    c9 = -0.30994571e-19
    p = (c0+Ta*(c1+Ta*(c2+Ta*(c3+Ta*(c4+Ta*(c5+Ta*(c6+Ta*(c7+Ta*(c8+Ta*(c9))))))))))
    Es = e_so/p**8
    Pv = RH/100 * Es * 100
    rho = (SP/(Rd*T))*(1-0.378*Pv/SP)
    return rho

def vindprod(vindspeed,Ta,RH,SP, cut_in,cut_out,A,Cp,n_gen):
    '''Tar inn vinddata og regner ut produksjon fra vindturbin, kW'''
    liste = []
    for i,v in enumerate(vindspeed):
        if v <= cut_in or v >= cut_out:
            P = 0
        else:
            rho = luft_tetthet(Ta[i],RH[i],SP[i])

            Pm = 0.5 * Cp * rho * A * v**3
            P = Pm * n_gen / 1000

        liste.append(P)
    return liste

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
#---Plot av fordeling av strømforbruket i løpet av ukedager---
import datetime
antal = np.zeros(24*7)
timeverdi = np.zeros(24*7)
tidspunkt = 'Mandag    Tirsdag    Onsdag    Torsdag    Fredag    Lørdag    Søndag'
for i in range(0,8760):
    datoen = dato[i]
    dateTimeInstance1 = datetime.datetime(int(datoen.split('.')[2]), int(datoen.split('.')[1]), int(datoen.split('.')[0]))
    ukedag = dateTimeInstance1.weekday()
    timen = time[i]
    forbruket = forbruk[i]
    uketime = ukedag*24+int(timen)
    antal[uketime] += 1
    timeverdi[uketime] += forbruket
for i,val in enumerate(timeverdi):
    if antal[i]>0:
        timeverdi[i] = val/antal[i]
plt.plot(timeverdi)
plt.xlabel(tidspunkt)
plt.ylabel('kWh/h')
plt.title('')
plt.show()

#%%
#---Beregning av kostnad---
#---Nettleie---
def nettleie(strømforbruk):
    # 0 = jan, 1 = feb, 2 = mar, 3 = apr, 4 = mai, 5 = jun, 6 = jul, 7 = aug, 8 = sep, 9 = okt, 10 = nov, 11 = des
    månedlig_strømforbruk = månedtot(strømforbruk)
    månedlig_makseffekt = månedmaks(strømforbruk)
    nettleie = [0,0,0,0,0,0,0,0,0,0,0,0]
    fastledd = 500 #kr/mnd
    effektledd_s = 32 #kr/kW/mnd april-september
    effektledd_v = 75 #kr/kW/mnd oktober-mars
    energiledd = 5 #øre/kWh
    elavgift_1 = 9.16 #øre/kWh januar-mars
    elavgift_2 = 15.84 #øre/kWh april-desember
    for i in range(0,12):
        nettleie[i] += fastledd + månedlig_strømforbruk[i] * energiledd/100
        if i > 2 and i < 9:
            nettleie[i] += effektledd_s * månedlig_makseffekt[i]
        else:
            nettleie[i] += effektledd_v * månedlig_makseffekt[i]
        if i < 3:
            nettleie[i] += månedlig_strømforbruk[i] * elavgift_1/100
        else:
            nettleie[i] += månedlig_strømforbruk[i] * elavgift_2/100
        nettleie[i] = round(nettleie[i],2)
    return nettleie

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
DoD = 0.8
E,V = 200,12.8 # Ah, V
kapasitet = 2.56 # kWh
antall = 10
tot_kap = kapasitet * antall
n_charge = 0.9
n_discharge = 0.9
C_charge = 1/7
C_discharge = 1/7
batterinivå_f,batterinivå_e = 0,0
batterinivå = []
ladestrøm = []
nytt_forbruk = []
for i, val in enumerate(time):
    if i < 24*365:
        timenr = int(val)

        if timenr >= 1 and timenr <= 6:
            #charge
            batterinivå_e = min(batterinivå_f + C_charge*tot_kap, tot_kap)
        elif timenr >= 17 and timenr <= 22:
            #discharge
            batterinivå_e = max(batterinivå_f - C_discharge*tot_kap, tot_kap*(1-DoD))
        
        batterinivå.append(batterinivå_e)
        # print(f'I time {timenr} er batterinivå {batterinivå_e}')
        ladestrøm.append(batterinivå_e - batterinivå_f)
        nytt_forbruk.append(forbruk[i]+batterinivå_e - batterinivå_f)
        batterinivå_f = batterinivå_e
plt.plot(døgnfordeling(nytt_forbruk,365))
plt.show()
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
plt.show()