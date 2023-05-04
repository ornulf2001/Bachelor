import numpy as np
import datetime
import matplotlib.pyplot as plt

#---Regruppering av data---
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
        start += 24*månedlengde[i]
    return månedliste

def døgnfordeling(liste):
    '''Viser fordelingen av i løpet av timene i døgnet'''
    time = np.zeros(24)
    døgn = int(len(liste)/24)
    for t in range(0,24):
        for d in range(0,døgn):
            time[t] += liste[d*24+t]
    time = time/døgn
    return time

#---Plot av fordeling av strømforbruket i løpet av ukedager---
def ukesfordeling(forbruk,dato,time):

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
    #return timeverdi
    plt.plot(timeverdi)
    plt.xlabel(tidspunkt)
    plt.ylabel('kWh/h')
    plt.title('')
    plt.show()

#---Maks effekt per måned---
def månedmaks(årsliste):
    '''Finner høyeste verdi for hver måned. Fin for å finne makseffekt'''
    måned_maks = [0,0,0,0,0,0,0,0,0,0,0,0]
    døgntot = 0
    for i,val in enumerate([31,28,31,30,31,30,31,31,30,31,30,31]):
        måned_maks[i] = max(årsliste[24*døgntot:24*(døgntot+val)])
        døgntot += val
    return måned_maks

#---Totalt forbruk per måned---
def månedtot(årsliste):
    '''Finner total verdi for hver måned. Fin for å finne strømforbruk'''
    måned_tot = [0,0,0,0,0,0,0,0,0,0,0,0]
    døgntot = 0
    for i,val in enumerate([31,28,31,30,31,30,31,31,30,31,30,31]):
        måned_tot[i] = round(sum(årsliste[24*døgntot:24*(døgntot+val)]),1)
        døgntot += val
    return måned_tot

# Definerer trigonometriske funksjoner for bruk med grader
sind = lambda degrees: np.sin(np.deg2rad(degrees))
cosd = lambda degrees: np.cos(np.deg2rad(degrees))
asind = lambda degrees: np.rad2deg(np.arcsin(degrees))
acosd = lambda degrees: np.rad2deg(np.arccos(degrees))

def solprod_eksperimentell(Gb_n, Gd_h, Ta, antal, Zs, beta):
    '''Tar inn fil med soldata, areal, og vinkler. Bruker dette
    til å regne ut produksjonen fra solenergi. Gitt som kWh/h. Om det er
    roterende panelstativ, sett Zs og beta til 666'''
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
            if Zs == 666 and beta == 666: #sjekker om det er roterende panelstativ. Da optimaliseres vinkelen.
                theta = 0
                beta = alfa
            else:
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

def solprod_2(Gb_n, Gd_h, Ta, antal, Zs, beta):
    '''Tar inn fil med soldata, areal, og vinkler. Bruker dette
    til å regne ut produksjonen fra solenergi. Gitt som kWh/h. Om det er
    roterende panelstativ, sett Zs og beta til 666'''
    # faste verdier
    L = 60.61318
    LL = 12.01088
    SL = 15
    n_sol = 0.20 # Virkningsgrad sol !!!
    LST = 0
    A = 1.755*1.038   # Areal per panel !!!
    # Tap pga. varme
    T_tap_Pmpp = -0.0035 #Varierer per paneltype, Temperaturkoefisient Pmpp
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

def nettleie(strømforbruk):
    '''Bruker liste for strømforbruket for å beregne nettleie for hver måned'''
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

def batteri(kap,forbruk,time_liste):
    '''Bruker batteri til å jamne ut strømforbruket'''
    DoD = 0.8
    E,V = 200,12.8 # Ah, V
    #kapasitet = 8*2.56 # kWh
    antall = 10
    tot_kap = kap# * antall
    n_charge = 0.9
    n_discharge = 0.9
    C_charge = 1/7
    C_discharge = 1/7
    batterinivå_f,batterinivå_e = 0,0
    batterinivå = []
    ladestrøm = []
    nytt_forbruk = []
    charging,discharging = False,False
    for i, val in enumerate(time_liste):
        if i < 24*365:
            timenr = int(val)
            strøm = 0

            if timenr >= 1 and timenr <= 6: charging = True
            elif forbruk[i] < 0:
                charging = True
                ikke_salg = True
            if timenr >= 17 and timenr <= 22: discharging = True

            if charging:
                #charge
                if ikke_salg:
                    opplading = -forbruk[i]
                    batterinivå_e = min(batterinivå_f + opplading, tot_kap)
                    strøm = (batterinivå_e - batterinivå_f)/n_charge
                else:
                    opplading = C_charge*tot_kap
                    batterinivå_e = min(batterinivå_f + opplading, tot_kap)
                    strøm = (batterinivå_e - batterinivå_f)/n_charge
            if discharging:
                #discharge
                utlading = C_discharge*tot_kap/n_discharge
                batterinivå_e = max(batterinivå_f - utlading, tot_kap*(1-DoD))
                strøm = (batterinivå_e - batterinivå_f)*n_discharge
            
            batterinivå.append(batterinivå_e)
            # print(f'I time {timenr} er batterinivå {batterinivå_e}')
            ladestrøm.append(strøm)
            nytt_forbruk.append(forbruk[i]+strøm)
            batterinivå_f = batterinivå_e
            charging,discharging,ikke_salg = False, False, False
    return nytt_forbruk

def strømkostnad(forbruk,strømpris_liste,spotpris_liste):
    '''Beregner kostnaden for strøm for hver time'''
    strømkostnad = []
    for i in range(0,8760):
        if forbruk[i]>=0:
            strømkostnad.append(forbruk[i]*strømpris_liste[i])
        else:
            strømkostnad.append(forbruk[i]*spotpris_liste[i])
    return strømkostnad
