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
import time

#---Priser og kostnad---
flis_pris = 0.4   # kr/kWh
#---Komponenter---
PV_panel = 2279
festeklemme = 29
festeskinne = 595
fast_stativ = 33500
rot_stativ = 100000
inverter = 31508      # 17 kW
sol_installasjon = 50000
vindturbin_h2 = 20270
installasjon_vind = 10000
fliskjele = 800000
installasjon_bio = 500000

# Sol
# Regner ut solproduksjon per panel
sol_sanitær = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 29, beta = 25)
sol_nedre_restaurant = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = -60, beta = 15)
sol_øvre_restaurant = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = -60, beta = 35)
sol_fastmontert = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 0, beta = 22)
sol_roterende = solprod_2(Gb_n, Gd_h, Ta, antal = 1, Zs = 666, beta = 666)

# Vind
# Regner ut vindproduksjon fra 1 vindturbin
vind_horisontal2 = vindprod(vindspeed,Ta,RH,SP, cut_in= 3, cut_out= 50, A = np.pi*2.35**2/4, Cp = 0.385, n_gen = 0.9)

#%%
#                                san, nedre, øvre, fast, rot, bio, batt, vind
def lønnsomhet(energikilde):# = [  1,     1,    1,    1,   1,   1,  100,    0]):
#                                0/1     0/1   0/1  fast/rot/0 0/1   kWh   antall
    '''Tar inn en liste med antall enheter av de ulike enerkikildene, returnerer NNV og installasjonskostnad'''
    
    #---Variabler---
    paneler_sanitær = 28             # antall
    paneler_nedre_restaurant = 20
    paneler_øvre_restaurant = 32
    paneler_fastmontert = 24
    paneler_roterende = 15

    vindturbiner_h2 = energikilde[7]

    bioandel = 0.21*energikilde[5]          # % av strømforbruket som kan dekkes av bio
    batterikapasitet = energikilde[6]       # kWh lagringskapasitet

    PV_tak_sanitær = festeskinne*36+festeklemme*162+PV_panel*paneler_sanitær+sol_installasjon
    PV_tak_nedre = festeskinne*15+festeklemme*66+PV_panel*paneler_nedre_restaurant+sol_installasjon
    PV_tak_øvre = festeskinne*24+festeklemme*102+PV_panel*paneler_øvre_restaurant+sol_installasjon
    PV_fri = fast_stativ+paneler_fastmontert*PV_panel+sol_installasjon
    PV_rot = rot_stativ+PV_panel*paneler_roterende+sol_installasjon

    batteribank = 4595/(12*260)*1000*batterikapasitet  #kr/kWh * kWh
    installasjon_batteri = 10000*min(1,energikilde[6])
    installasjonskostnader_batt = batteribank + installasjon_batteri
    #---Installasjon---
    pris_sol = PV_tak_sanitær*energikilde[0] + PV_tak_nedre*energikilde[1] + PV_tak_øvre*energikilde[2] + PV_fri*energikilde[3] + PV_rot*energikilde[4]
    pris_invertere = inverter*(energikilde[0]+energikilde[1]+energikilde[2]+min(1,int((energikilde[3]+1)/2))+min(1,energikilde[4])+min(1,energikilde[6])+min(1,energikilde[7]))

    installasjonskostnader = pris_sol + pris_invertere + installasjonskostnader_batt + energikilde[5]*(fliskjele+installasjon_bio) + vindturbin_h2*energikilde[7]+installasjon_vind*min(1,energikilde[7])
    # Regner ut solproduksjon
    
    total_solproduksjon=[]
    solanlegg = [sol_sanitær, sol_nedre_restaurant, sol_øvre_restaurant, sol_fastmontert, sol_roterende]
    sol_antall = [paneler_sanitær*energikilde[0], paneler_nedre_restaurant*energikilde[1], paneler_øvre_restaurant*energikilde[2], paneler_fastmontert*energikilde[3], paneler_roterende*energikilde[4]]
    for i in range(0,8760):
        sol_prod_time = 0
        for a,anlegg in enumerate(solanlegg):
            sol_prod_time += anlegg[i]*sol_antall[a]
        total_solproduksjon.append(sol_prod_time)

    # ---Vind---
    total_vindproduksjon=[num*vindturbiner_h2 for num in vind_horisontal2]

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

    #---Beregning av kostnad---
    #---Nettleie---

    nettleie_kr = nettleie(energibalanse_batt)
    #---Strømkostnad---
    strømkostnaden = strømkostnad(energibalanse_batt,strømpris_liste,spotpris_liste)
    #---Total kostnad---
    total_årlig_kostnad_etter = 1.25*round(sum(strømkostnaden)+49*12 + sum(nettleie_kr) + sum(flis_energi)*flis_pris)   # + installasjonskostnader# + innstallasjonskostnad/levetid ? + vedlikehold
    total_årlig_kostnad_før = 1.25*round(sum(strømkostnad(energiforbruk_liste,strømpris_liste,spotpris_liste))+49*12+sum(nettleie(energiforbruk_liste)))

    NNVf,NNVe = 0,-installasjonskostnader
    r = 0.05
    for i in range(1,31):
        NNVf += -total_årlig_kostnad_før/(1+r)**i
        NNVe += -total_årlig_kostnad_etter/(1+r)**i
    for i in range(1,11):
        NNVe += -installasjonskostnader_batt/(1+r)**(3*i)
    for i in range(1,4):
        NNVe += -pris_invertere/(1+r)**(10*i)



    return NNVe,installasjonskostnader,min(energibalanse_batt)

def lønnsomhet_stats(energikilde):# = [  1,     1,    1,    1,   1,   1,  100,    0]):
#                                     0/1     0/1   0/1  fast/rot/0 0/1   kWh   antall
    '''Tar inn en liste med antall enheter av de ulike enerkikildene, printer relevant info'''
    
    #---Variabler---
    paneler_sanitær = 28             # antall
    paneler_nedre_restaurant = 20
    paneler_øvre_restaurant = 32
    paneler_fastmontert = 24
    paneler_roterende = 15

    vindturbiner_h2 = energikilde[7]

    bioandel = 0.21*energikilde[5]          # % av strømforbruket som kan dekkes av bio
    batterikapasitet = energikilde[6]       # kWh lagringskapasitet

    PV_tak_sanitær = festeskinne*36+festeklemme*162+PV_panel*paneler_sanitær+sol_installasjon
    PV_tak_nedre = festeskinne*15+festeklemme*66+PV_panel*paneler_nedre_restaurant+sol_installasjon
    PV_tak_øvre = festeskinne*24+festeklemme*102+PV_panel*paneler_øvre_restaurant+sol_installasjon
    PV_fri = fast_stativ+paneler_fastmontert*PV_panel+sol_installasjon
    PV_rot = rot_stativ+PV_panel*paneler_roterende+sol_installasjon

    batteribank = 4595/(12*260)*1000*batterikapasitet  #kr/kWh * kWh
    installasjon_batteri = 10000*min(1,energikilde[6])
    installasjonskostnader_batt = batteribank + installasjon_batteri
    #---Installasjon---
    pris_sol = PV_tak_sanitær*energikilde[0] + PV_tak_nedre*energikilde[1] + PV_tak_øvre*energikilde[2] + PV_fri*energikilde[3] + PV_rot*energikilde[4]
    pris_invertere = inverter*(energikilde[0]+energikilde[1]+energikilde[2]+min(1,int((energikilde[3]+1)/2))+min(1,energikilde[4])+min(1,energikilde[6])+min(1,energikilde[7]))

    installasjonskostnader = pris_sol + pris_invertere + installasjonskostnader_batt + energikilde[5]*(fliskjele+installasjon_bio) + vindturbin_h2*energikilde[7]+installasjon_vind*min(1,energikilde[7])
    # Regner ut solproduksjon
    
    total_solproduksjon=[]
    solanlegg = [sol_sanitær, sol_nedre_restaurant, sol_øvre_restaurant, sol_fastmontert, sol_roterende]
    sol_antall = [paneler_sanitær*energikilde[0], paneler_nedre_restaurant*energikilde[1], paneler_øvre_restaurant*energikilde[2], paneler_fastmontert*energikilde[3], paneler_roterende*energikilde[4]]
    for i in range(0,8760):
        sol_prod_time = 0
        for a,anlegg in enumerate(solanlegg):
            sol_prod_time += anlegg[i]*sol_antall[a]
        total_solproduksjon.append(sol_prod_time)

    # ---Vind---
    total_vindproduksjon=[num*vindturbiner_h2 for num in vind_horisontal2]

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

    #---Beregning av kostnad---
    #---Nettleie---

    nettleie_kr = nettleie(energibalanse_batt)
    #---Strømkostnad---
    strømkostnaden = strømkostnad(energibalanse_batt,strømpris_liste,spotpris_liste)
    #---Total kostnad---
    total_årlig_kostnad_etter = 1.25*round(sum(strømkostnaden)+49*12 + sum(nettleie_kr) + sum(flis_energi)*flis_pris)   # + installasjonskostnader# + innstallasjonskostnad/levetid ? + vedlikehold
    total_årlig_kostnad_før = 1.25*round(sum(strømkostnad(energiforbruk_liste,strømpris_liste,spotpris_liste))+49*12+sum(nettleie(energiforbruk_liste)))

    NNVf,NNVe = 0,-installasjonskostnader
    r = 0.05
    for i in range(1,31):
        NNVf += -total_årlig_kostnad_før/(1+r)**i
        NNVe += -total_årlig_kostnad_etter/(1+r)**i
    for i in range(1,11):
        NNVe += -installasjonskostnader_batt/(1+r)**(3*i)
    for i in range(1,4):
        NNVe += -pris_invertere/(1+r)**(10*i)
    
    #---Plot---

    plt.plot(døgnfordeling(energiforbruk_liste))
    plt.plot(døgnfordeling(energibalanse))
    plt.plot(døgnfordeling(energibalanse_batt))
    plt.show()

    print(f'NNV før:   {NNVf}\nNNV etter: {NNVe}')


    print(f'Total årlig kostnad før {total_årlig_kostnad_før} kr/år inkl. MVA')
    print(f'Total årlig kostnad etter {total_årlig_kostnad_etter} kr/år inkl. MVA')
    print(f'Kostnadsdifferanse: {total_årlig_kostnad_før-total_årlig_kostnad_etter}')
    print(f'\nInstallasjonskostnader: {installasjonskostnader}')
    print(f'Invertere: {pris_invertere}')
    print(f'Pris sol: {pris_sol}')
    print(f'Nettleie: {sum(nettleie(energiforbruk_liste))}---{sum(nettleie(energibalanse_batt))}')
    print(f'Strømforbruk: {round(sum(energibalanse_batt))}   før: {round(sum(energiforbruk_liste))}'
        f'\nKjøpt strøm:  {round(sum(kjøpt_strøm))}'
        f'\nSolgt strøm:    {round(sum(solgt_strøm))}')
    print(f'\nBio\n'
        f'\tInstallasjon/år: {energikilde[5]*(fliskjele+installasjon_bio)/30}\n'
        f'\tKostnad flis:    {sum(flis_energi)*flis_pris}\n'
        f'\tBespart:         {sum(strømkostnad(levert_energi,strømpris_liste,spotpris_liste))}')
    print(f'Største overproduksjon: {min(energibalanse_batt)}')
    



#%%
start = time.time()
NNVe_best = -1000000000
scenario_best = []
NNVe_nestbest = -1000000000
scenario_nestbest = []
scenarioer = {'test': -1000000000}
for sanitær in [0,1]:
    for nedre in [0,1]:
        for øvre in [0,1]:
            for fast in range(0,9): # maks 8
                for rot in range(0,4): # maks 3
                    if rot != 0:
                        fast = 0
                    for bio in [0,1]:
                        for batt in [0]:
                            for vind in [0]:
                                energikilder = [sanitær,nedre,øvre,fast,rot,bio,batt,vind]
                                scenario = lønnsomhet(energikilder)
                                NNVe = scenario[0]
                                string = str(energikilder)
                                if scenario[2]>-50:
                                    scenarioer[string] = scenario[0]
                                    # print(scenarioer)
                                    scenarioer = sorted(scenarioer.items(), key=lambda x:x[1], reverse = True)
                                    # print(scenarioer)
                                    if len(scenarioer)>10:
                                        # print(scenarioer)
                                        scenarioer.pop()
                                    scenarioer = dict(scenarioer)

                                # if scenario[0] > NNVe_best:
                                #     NNVe_nestbest = NNVe_best
                                #     scenario_nestbest = scenario_best
                                #     NNVe_best = scenario[0]
                                #     scenario_best = energikilder
                                # elif scenario[0] > NNVe_nestbest:
                                #     NNVe_nestbest = scenario[0]
                                #     scenario_nestbest = energikilder

# print(f'Beste kombinasjon er: {scenario_best}')
# print(f'Det gir NNV på {round(NNVe_best)}')
# print(f'Beste kombinasjon er: {scenario_nestbest}')
# print(f'Det gir NNV på {round(NNVe_nestbest)}')
stop = time.time()
print(f'Tid: {stop-start}sek')

print(scenarioer)



#                 san, nedre, øvre, fast, rot, bio, batt, vind
# print(lønnsomhet([1,     1,    1,    1,   1,   1,  100,  0])[1])
#%%
    
best_scenario = list(scenarioer)[0]
best_NNV = list(scenarioer.values())[0]
print(f'Beste scenario er {best_scenario}')

kun_tak = [1,1,1,0,0,0,0,0]
kun_fast = [0,0,0,1,0,0,0,0]
kun_rot = [0,0,0,0,1,0,0,0]
best = [1,1,1,8,0,1,0,0]
test = [0,0,0,0,3,0,0,0]
lønnsomhet_stats(best)


#%%
liste = [1,2,3]
liste2 = [2*num for num in liste]
print(liste2)
#%%
footballers_goals = {'Eusebio': 120, 'Cruyff': 104, 'Pele': 150, 'Ronaldo': 132, 'Messi': 125}
print(footballers_goals)
footballers_goals['Test'] = 0

footballers_goals = sorted(footballers_goals.items(), key=lambda x:x[1])
footballers_goals.pop()
footballers_goals.pop()
print(footballers_goals)