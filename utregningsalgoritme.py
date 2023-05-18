#%%
# --- Laster inn lister og verdier ---
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
# --- Laster inn kostnader og regner ut produksjon fra sol og vind ---
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
# --- Definerer inn funksjoner for lønnsomhet ---
#                                san, nedre, øvre, fast, rot, bio, LAB, LIB, vind
def lønnsomhet(energikilde):# = [  1,     1,    1,    1,   1,   1,  100, 100,   0]):
#                                0/1     0/1   0/1  fast/rot/0 0/1   kWh  kWh antall
    '''Tar inn en liste med antall enheter av de ulike enerkikildene, returnerer NPV og installasjonskostnad'''
    
    #---Variabler---
    paneler_sanitær = 48             # antall
    paneler_nedre_restaurant = 20
    paneler_øvre_restaurant = 32
    paneler_fastmontert = 24
    paneler_roterende = 15

    vindturbiner_h2 = energikilde[8]

    bioandel = 0.21*energikilde[5]          # % av strømforbruket som kan dekkes av bio
    LAB_kapasitet = energikilde[6]       # kWh lagringskapasitet
    LIB_kapasitet = energikilde[7]       # kWh lagringskapasitet

    PV_tak_sanitær = festeskinne*36+festeklemme*162+PV_panel*paneler_sanitær+sol_installasjon
    PV_tak_nedre = festeskinne*15+festeklemme*66+PV_panel*paneler_nedre_restaurant+sol_installasjon-0.5*sol_installasjon*max(0,(energikilde[1]+energikilde[2])-1)
    PV_tak_øvre = festeskinne*24+festeklemme*102+PV_panel*paneler_øvre_restaurant+sol_installasjon
    PV_fri = fast_stativ+paneler_fastmontert*PV_panel+sol_installasjon
    PV_rot = rot_stativ+PV_panel*paneler_roterende+sol_installasjon

    LAB_pris = 4595/(12*260)*1000*LAB_kapasitet  #kr/kWh * kWh
    installasjon_LAB = 10000*min(1,energikilde[6])
    installasjonskostnader_LAB = LAB_pris + installasjon_LAB
    LIB_pris = 4200*LIB_kapasitet  #kr/kWh * kWh
    installasjon_LIB = LIB_pris*0.2*min(1,energikilde[7])
    installasjonskostnader_LIB = LIB_pris + installasjon_LIB
    #---Installasjon---
    pris_sol = PV_tak_sanitær*energikilde[0] + PV_tak_nedre*energikilde[1] + PV_tak_øvre*energikilde[2] + PV_fri*energikilde[3] + PV_rot*energikilde[4]
    pris_invertere = inverter*(energikilde[0]+min((energikilde[1]+energikilde[2]),1)+min(1,int((energikilde[3]+1)/2))+min(1,energikilde[4])+min(1,energikilde[6])+min(1,energikilde[7])+min(1,energikilde[8]))

    installasjonskostnader = pris_sol + pris_invertere + installasjonskostnader_LAB + installasjonskostnader_LIB + energikilde[5]*(fliskjele+installasjon_bio) + vindturbin_h2*energikilde[8]+installasjon_vind*min(1,energikilde[8])
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

    #---batteri---
    if energikilde[6] != 0:
        energibalanse_batt = batteri(LAB_kapasitet,energibalanse,time_liste)
    elif energikilde[7] != 0:
        energibalanse_batt = batteri_2(LIB_kapasitet,energibalanse,time_liste,0.95)
    else:
        energibalanse_batt = energibalanse

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

    # NPVf,NPVe = 0,-installasjonskostnader
    # r = 0.05
    # for i in range(1,31):
    #     NPVf += -total_årlig_kostnad_før/(1+r)**i
    #     NPVe += -total_årlig_kostnad_etter/(1+r)**i
    # for i in range(1,11):
    #     NPVe += -installasjonskostnader_LAB/(1+r)**(3*i)
    # for i in range(1,4):
    #     NPVe += -(pris_invertere+installasjonskostnader_LIB)/(1+r)**(10*i)

    NPVf,NPVe = 0,-installasjonskostnader
    r = 0.05
    for i in range(1,31):
        NPVf += -0/(1+r)**i
        NPVe += (total_årlig_kostnad_før-total_årlig_kostnad_etter)/(1+r)**i
    for i in range(1,11):
        NPVe += -installasjonskostnader_LAB/(1+r)**(3*i)
    for i in range(1,4):
        NPVe += -(pris_invertere+installasjonskostnader_LIB)/(1+r)**(10*i)



    return NPVe,installasjonskostnader,min(energibalanse_batt),energibalanse_batt,total_solproduksjon,levert_energi

def lønnsomhet_stats(energikilde):# = [  1,     1,    1,    1,   1,   1,  100, 100,   0]):
#                                     0/1     0/1   0/1  fast/rot/0 0/1   kWh  kWh  antall
    '''Tar inn en liste med antall enheter av de ulike enerkikildene, printer relevant info'''
    
    #---Variabler---
    paneler_sanitær = 48             # antall
    paneler_nedre_restaurant = 20
    paneler_øvre_restaurant = 32
    paneler_fastmontert = 24
    paneler_roterende = 15

    vindturbiner_h2 = energikilde[8]

    bioandel = 0.21*energikilde[5]          # % av strømforbruket som kan dekkes av bio
    LAB_kapasitet = energikilde[6]       # kWh lagringskapasitet
    LIB_kapasitet = energikilde[7]       # kWh lagringskapasitet

    PV_tak_sanitær = festeskinne*36+festeklemme*162+PV_panel*paneler_sanitær+sol_installasjon
    PV_tak_nedre = festeskinne*15+festeklemme*66+PV_panel*paneler_nedre_restaurant+sol_installasjon-0.5*sol_installasjon*max(0,(energikilde[1]+energikilde[2])-1)
    PV_tak_øvre = festeskinne*24+festeklemme*102+PV_panel*paneler_øvre_restaurant+sol_installasjon
    PV_fri = fast_stativ+paneler_fastmontert*PV_panel+sol_installasjon
    PV_rot = rot_stativ+PV_panel*paneler_roterende+sol_installasjon

    LAB_pris = 4595/(12*260)*1000*LAB_kapasitet  #kr/kWh * kWh
    installasjon_LAB = 10000*min(1,energikilde[6])
    installasjonskostnader_LAB = LAB_pris + installasjon_LAB
    LIB_pris = 4200*LIB_kapasitet  #kr/kWh * kWh
    installasjon_LIB = LIB_pris*0.2*min(1,energikilde[7])
    installasjonskostnader_LIB = LIB_pris + installasjon_LIB
    #---Installasjon---
    pris_sol = PV_tak_sanitær*energikilde[0] + PV_tak_nedre*energikilde[1] + PV_tak_øvre*energikilde[2] + PV_fri*energikilde[3] + PV_rot*energikilde[4]
    pris_invertere = inverter*(energikilde[0]+min((energikilde[1]+energikilde[2]),1)+min(1,int((energikilde[3]+1)/2))+min(1,energikilde[4])+min(1,energikilde[6])+min(1,energikilde[7])+min(1,energikilde[8]))

    installasjonskostnader = pris_sol + pris_invertere + installasjonskostnader_LAB + installasjonskostnader_LIB + energikilde[5]*(fliskjele+installasjon_bio) + vindturbin_h2*energikilde[8]+installasjon_vind*min(1,energikilde[8])
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

    #---batteri---
    if energikilde[6] != 0:
        energibalanse_batt = batteri(LAB_kapasitet,energibalanse,time_liste)
    elif energikilde[7] != 0:
        energibalanse_batt = batteri_2(LIB_kapasitet,energibalanse,time_liste,0.95)
    else:
        energibalanse_batt = energibalanse

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

    # NPVf,NPVe = 0,-installasjonskostnader
    # r = 0.05
    # for i in range(1,31):
    #     NPVf += -total_årlig_kostnad_før/(1+r)**i
    #     NPVe += -total_årlig_kostnad_etter/(1+r)**i
    # for i in range(1,11):
    #     NPVe += -installasjonskostnader_LAB/(1+r)**(3*i)
    # for i in range(1,4):
    #     NPVe += -(pris_invertere+installasjonskostnader_LIB)/(1+r)**(10*i)
    NPVf,NPVe = 0,-installasjonskostnader
    r = 0.05
    for i in range(1,31):
        NPVf += -0/(1+r)**i
        NPVe += (total_årlig_kostnad_før-total_årlig_kostnad_etter)/(1+r)**i
    for i in range(1,11):
        NPVe += -installasjonskostnader_LAB/(1+r)**(3*i)
    for i in range(1,4):
        NPVe += -(pris_invertere+installasjonskostnader_LIB)/(1+r)**(10*i)
    
    #---Plot---

    plt.plot(døgnfordeling(energiforbruk_liste), label = 'Referansesystem')
    # plt.plot(døgnfordeling(energibalanse))
    plt.plot(døgnfordeling(energibalanse_batt), label = 'Mikronett')
    plt.xlabel('Time')
    plt.ylabel('Forbruk fra strømnett [kWh]')
    plt.legend()
    plt.show()

    print(f'NPV før:   {NPVf} kr\nNPV etter: {round(NPVe,2)} kr')


    print(f'Total årlig kostnad før {total_årlig_kostnad_før} kr/år inkl. MVA')
    print(f'Total årlig kostnad etter {total_årlig_kostnad_etter} kr/år inkl. MVA')
    print(f'Kostnadsdifferanse: {total_årlig_kostnad_før-total_årlig_kostnad_etter} kr')
    print(f'Nedbetalingstid: {round(installasjonskostnader/(total_årlig_kostnad_før-total_årlig_kostnad_etter),2)} år')
    print(f'\nInstallasjonskostnader: {installasjonskostnader} kr')
    print(f'Kostnad invertere: {pris_invertere}')
    print(f'Nettleie: {round(sum(nettleie(energibalanse_batt)),2)} kr før: {sum(nettleie(energiforbruk_liste))} kr')
    print(f'Strømforbruk: {round(sum(energibalanse_batt))} kWh  før: {round(sum(energiforbruk_liste))} kWh'
        f'\nKjøpt strøm:  {round(sum(kjøpt_strøm))} kWh'
        f'\nSolgt strøm:   {round(sum(solgt_strøm))} kWh')
    print(f'Største overproduksjon: {round(min(energibalanse_batt),2)} kWh/h')
    print(f'\nBio\n'
          f'\tInstallasjonskostnad: {round(fliskjele+installasjon_bio,2)} kr\n'
        f'\tLevert energi:   {round(sum(levert_energi),2)} kWh\n'
        f'\tEnergi flis:     {round(sum(flis_energi),2)} kWh\n'
        f'\tKostnad flis:    {round(sum(flis_energi)*flis_pris,2)} kr\n'
        f'\tMengde flis:     {round(sum(Vol_flis),2)} lm^3\n'
        f'\n\tBio-andel: {round(sum(levert_energi)/sum(energiforbruk_liste),3)}')
    print(f'\nSol\n'
          f'\tInstallasjonskostnad: {pris_sol} kr\n'
          f'\tProdusert energi: {round(sum(total_solproduksjon),2)} kWh\n'
          f'\tSol-andel: {round(sum(total_solproduksjon)/sum(energiforbruk_liste),3)}')



#%%
# --- For løkke for å finne scenarioet som har høyest NPV ---
start = time.time()
NPVe_best = -1000000000
scenario_best = []
NPVe_nestbest = -1000000000
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
                        for LAB in [0]:
                            for LIB in [0]:
                                if LAB != 0:
                                    LIB = 0
                                for vind in [0]:
                                    energikilder = [sanitær,nedre,øvre,fast,rot,bio,LAB,LIB,vind]
                                    scenario = lønnsomhet(energikilder)
                                    NPVe = scenario[0]
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

                                # if scenario[0] > NPVe_best:
                                #     NPVe_nestbest = NPVe_best
                                #     scenario_nestbest = scenario_best
                                #     NPVe_best = scenario[0]
                                #     scenario_best = energikilder
                                # elif scenario[0] > NPVe_nestbest:
                                #     NPVe_nestbest = scenario[0]
                                #     scenario_nestbest = energikilder

# print(f'Beste kombinasjon er: {scenario_best}')
# print(f'Det gir NPV på {round(NPVe_best)}')
# print(f'Beste kombinasjon er: {scenario_nestbest}')
# print(f'Det gir NPV på {round(NPVe_nestbest)}')
stop = time.time()
print(f'Tid: {stop-start}sek')

print(scenarioer)

#%%
# --- Plotter statistikk for scenario ---

kun_tak = [1,1,1,0,0,0,0,0,0]
kun_fast = [0,0,0,1,0,0,0,0,0]
kun_rot = [0,0,0,0,1,0,0,0,0]
flis = [0,0,0,0,0,1,0,0,0]
rang_1 = [1,0,1,4,0,1,0,0,0]
rang_2 = [0,0,1,6,0,1,0,0,0]
rang_3 = [1,0,0,5,0,1,0,0,0]
rang_4 = [0,0,0,7,0,1,0,0,0]
rang_5 = [1,1,1,3,0,1,0,0,0]
rang_6 = [0,1,1,5,0,1,0,0,0]
rang_7 = [1,1,0,4,0,1,0,0,0]
rang_8 = [0,1,0,6,0,1,0,0,0]
test = [0,0,0,0,0,0,0,0,0]

lønnsomhet_stats(flis)
#%%
liste = rang_1
plt.plot(månedtot(lønnsomhet(liste)[3]))
plt.plot(månedtot(lønnsomhet(liste)[4]))
plt.plot(månedtot(lønnsomhet(liste)[5]))
plt.show()
print(f'Reduksjon i strømimport: {round(sum(energiforbruk_liste)-sum(lønnsomhet(liste)[3]))}'
      f'\n\tEnergi fra bio: {round(sum(lønnsomhet(liste)[5]))}'
      f'\n\tEnergi fra sol: {round(sum(lønnsomhet(liste)[4]))}'
      f'\n\tBio-andel: {round(sum(lønnsomhet(liste)[5])/sum(energiforbruk_liste),3)}'
      f'\n\tSol-andel: {round(sum(lønnsomhet(liste)[4])/sum(energiforbruk_liste),3)}')

#%%
liste = [1,2,3]
liste2 = [2*num for num in liste]
print(liste2)