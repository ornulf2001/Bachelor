#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
# --- Strømforbruk ---

fil = 'Verdier Bacheloroppgave - Timesdata strøm.csv'
df = pd.read_csv(fil, sep=',')
juli = df.iloc[:]['Verdi juli']
august = df.iloc[:]['Verdi august']
september = df.iloc[:]['Verdi september']
def timezwapper(måned,døgn):
    måned_dag = np.zeros(døgn)
    for d in range(0,døgn):
        for t in range(0,24):
            måned_dag[d] += måned[d*24+t]
    return måned_dag

juli_dag = timezwapper(juli,31)
august_dag = timezwapper(august,31)
september_dag = timezwapper(september,30)
# plt.plot(juli_dag)
# plt.show()
# plt.plot(august_dag)
# plt.show()
# plt.plot(september_dag)
# plt.show()
#print(juli_dag)
#print(sum(juli_dag))
#print(sum(juli))
def stats(måned):
    return str(round(np.sum(måned),2)) + '\t\t' + str(round(np.mean(måned),2)) + '\n'
#print(stats(juli_dag))
print('Sum per måned: \t\tAvg per dag:\n'+ 'Juli:\t\t' + stats(juli_dag) + 'August:\t\t' + stats(august_dag) + 'September:\t' + stats(september_dag))
def døgnfordeling(måned,døgn):
    time = np.zeros(24)
    for t in range(0,24):
        for d in range(0,døgn):
            time[t] += måned[d*24+t]
    time = time/døgn
    return time
#print(døgnfordeling(juli,31))
plt.plot(døgnfordeling(juli,31))
plt.show()

#%%
fil_2 = 'Timesforbruk hele 2021 og 2022 V2.csv'
fil_3 = 'Timesforbruk hele 2021 og 2022.csv'
df = pd.read_csv(fil_3, sep=';')
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
døgnfordelt = døgnfordeling(forbruk,int(len(forbruk)/24))
print(f'Totalt strømforbruk: {sum(forbruk)}')

plt.plot(døgnfordelt)
plt.show()
# time
# int(len(forbruk)/24)
# forbruk

#%%
# --- Automagisk melding i Teams :) ---

import pymsteams
for i in [1,2,3,4,5]:
    myTeamsMessage = pymsteams.connectorcard("https://studntnu.webhook.office.com/webhookb2/c66adcd5-80ed-4d70-871d-5b1134427fa0@09a10672-822f-4467-a5ba-5bb375967c05/IncomingWebhook/d66ea2b82c734381a67d369368135444/f0cb5ccb-99e0-4d24-ba80-42cd05c88c63")
    myTeamsMessage.text(f"Hei {i}. gang")
    myTeamsMessage.send()
myTeamsMessage = pymsteams.connectorcard("https://studntnu.webhook.office.com/webhookb2/c66adcd5-80ed-4d70-871d-5b1134427fa0@09a10672-822f-4467-a5ba-5bb375967c05/IncomingWebhook/d66ea2b82c734381a67d369368135444/f0cb5ccb-99e0-4d24-ba80-42cd05c88c63")
myTeamsMessage.text(f"Takk for meeej")
myTeamsMessage.send()

#%%
# --- Sol og vind ---

# Laster inn fil
fil_tmy = 'tmy-data_modifisert.csv'
df = pd.read_csv(fil_tmy, sep=',')

# Egen liste for hver 
G_h = df.iloc[:]['G(h)']   # Global horizontal irradiance
Gb_n = df.iloc[:]['Gb(n)'] # Direct (beam) irradiance
Gd_h = df.iloc[:]['Gd(h)'] # Diffuse horizontal irradiance
Ta = df.iloc[:]['T2m']     # Dry bulb (air) temperature
vindspeed = df.iloc[:]['WS10m'] # Windspeed

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
    n_sol = 0.20 # Virkningsgrad sol !!!
    LST = -1
    A = 1.8   # Areal per panel !!!
    # Tap pga. varme
    T_tap_Pmpp = -0.0045 #Varierer per paneltype, Temperaturkoefisient Pmpp
    T_noct = 46          #Varierer per paneltype, Celletemp test
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
            AST = LST + ET/60 - 4/60*(SL-LL) # (-) når østlig halvkule
            h = (AST - 12) * 15
            alfa = asind(sind(L)*sind(delta)+cosd(L)*cosd(delta)*cosd(h))
            z = asind(cosd(delta)*sind(h)/cosd(alfa))
            theta = acosd(sind(L)*sind(delta)*cosd(beta)-cosd(L)*sind(delta)*sind(beta)*cosd(Zs)
                        +cosd(L)*cosd(delta)*cosd(h)*cosd(beta)+sind(L)*cosd(delta)*sind(beta)*cosd(h)*cosd(Zs)
                        +cosd(delta)*sind(h)*sind(beta)*sind(Zs))
            
            G = Gb_n[i] * cosd(theta) + Gd_h[i] * (180 - beta)/180
            if G < 0: G = 0
            P = G * n_sol

            Tc = (T_noct-T_a_noct)*(G/Gt_noct)*(1-n_sol/ta)+Ta[i]
            tap_varme = T_tap_Pmpp/(Tc-T_noct)

            produksjon = (P + tap_varme) * antal
            
            # if i < 48:
            #     print(f'N = {N} for dato {df.iloc[i][0]}, LST = {LST}, delta = {delta}, B = {B}, ET = {ET}, AST = {AST}, h = {h}, alfa = {alfa}')
            test_list.append(produksjon)
    return test_list



# Regner ut solproduksjon
tak_1 = solprod(Gb_n, Gd_h, Ta, antal = 1, Zs = 0, beta = 20)
print(sum(tak_1))
print(max(tak_1))

# ---plotting av graf
# plt.plot(tak_1[0:8760])
# plt.show()

# %%

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
        print(slutt)
        start += 24*månedlengde[i]
    return månedliste
