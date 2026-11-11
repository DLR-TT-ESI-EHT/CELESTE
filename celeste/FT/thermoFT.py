import numpy as np
import pandas as pd
import numba as nb

from ..CSL import thermo as th

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

'''
1. Thermo: calculation of MW, LHV cp, h, s, g of a single component or stream fir FT components
2. ThermoExergy: calculation of the stream exergy considering the physical and chemical exergy of the stream
'''

@nb.njit
def _VapCalc(T, Tmin, Tmax, B, Tc, n, cpl):
    '''
    need to test
    '''

    N = len(Tmin)
    Vap = np.empty((2,N))

    for i in range(N):
        if T>Tmax[i]:
            Vap[0,i] = 0
        elif np.logical_and(T>=Tmin[i], T<=Tmax[i]):
            Vap[0,i] = B[i]*(1-T/Tc[i])*n[i]
        elif T<Tmin[i]:
            Vap[0,i] = B[i]*(1-Tmin[i]/Tc[i])**n[i] + cpl[i]*(T-Tmin[i])

        Vap[1,i] = Vap[0,i]/T

    return Vap

@nb.njit
def _phaseVap(phaseID, T, Tmin, Tmax, B, Tc, n, cpl, NHC):
    if phaseID == 1:
        VAP = _VapCalc(T, Tmin, Tmax, B, Tc, n, cpl)
    else:
        VAP = np.zeros((2, NHC))
    return VAP

@nb.njit
def _mtCalcStatic(m, c, thermArr, CNumberi):
    therm_sep = np.empty((3,len(CNumberi)))
    for i in range(3):
        for l in range(len(CNumberi)):
            therm = 0
            for j in range(5):
                therm += m[i,j]*thermArr[j]*CNumberi[l] + c[i,j]*thermArr[j]
            therm_sep[i,l] = therm
    return therm_sep

@nb.njit
def _thermoArrStat(t):
    return np.array([t*t*t*t, t*t*t, t*t, t, 1.]) 

class Thermo:  
    file_path = files("celeste.data").joinpath("Hydrocarbons_ChemProp.csv").as_posix()
    df = pd.read_csv(file_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    df = df.set_index("Formula")
    R = 8.314462618e-3
    
    def __init__(self, species, T, Xi, phase='Gas', P=1., T_unit='K'):
    
        self.species = species
        self._phase = phase  

        self._phaseID = 1 if phase == 'liquid' or phase == 'Liquid' else 0

        # separate NASA and HC components
        self._spFuelCH4, self._spHC = self.getSpecies() 
        
        if T_unit in ['C', 'c']:
            T = T + 273.15
        
        self._T = T 
        
        self._Xi = Xi
        self._P = P
        self.__NFuelCH4 = len(self._spFuelCH4)
        self.__NHC = len(self._spHC)

        self.__DATA = self.df.loc[self.species] 
        ## HC thermo data
        self.__DATA_HC = self.df.loc[self._spHC] 
        ## LHV at standard conditions
        '''
        from C1 to C4 --> gas phase
        from C5 to C15 --> liquid phase
        '''
        self.__DATA['lhv'] = self.__DATA[f'LHV_kJ/mol']   
        self._MWi = self.__DATA['MW'].values
        
        self.HNumberi = self.__DATA['H_N'].values 
        
        self.__HVAP_B = self.__DATA_HC['HVAP_B'].values 
        self.__HVAP_n = self.__DATA_HC['HVAP_n'].values #, dtype=object)
        self.__HVAP_Tb =self.__DATA_HC['T_boil_K'].values
        self.__HVAP_Tc = self.__DATA_HC['Crit_T_K'].values
        self.__HVAP_Tmin = self.__DATA_HC['HVAP_Tmin_K'].values
        self.__HVAP_Tmax = self.__DATA_HC['HVAP_Tmax_K'].values
     
        self.__cpl = self.__DATA_HC['cp_std_liq_kJ/mol*K'].values
        self.CNumberi = self.__DATA_HC['C_N'].values
        
        self.__m = np.array([
            [1.60208009e-13, -3.85369069e-10, 2.84263828e-7, -2.1076165e-5, 0.0126558917],
            [-2.03518749e-11, 2.39474418e-8, 2.18318243e-5, 5.40208647e-3,-24.959112],
            [-3.21627106e-14, 5.28128653e-11,-4.63466461e-8, 9.28472733e-5, 0.0147243196]
        ])
        self.__c = np.array([
            [-3.32104578e-13, 7.15023059e-10, -4.77027078e-7, 1.21672696e-4, -0.00329707471],
            [2.12326032e-11,-1.2232199e-8,-6.26259659e-6, 1.17389703e-2,- 44.6550556],
            [-6.42129853e-14, 1.91117309e-10,-1.89411673e-7, 9.19937061e-5,0.136713671]
        ])

        self.thermoArr()      
        
        if self.__NFuelCH4 > 0:
            self.__FuelCH4_Th = th.Thermo(self._spFuelCH4, self._T, np.zeros((self.__NFuelCH4)), self._P) 
        else:
            self.__FuelCH4_Th = th.ThermoDum(self._T)    
        
        if self.__NHC > 0:
            self.calcThermoHC()
        else: # temporary
            self.__HC_cp, self.__HC_h, self.__HC_s, self.__HC_g = np.array([]), np.array([]), np.array([]),np.array([])
        
        self.calcThermoSep()
        self.calcThermoMix()
        
    
    ## Methods 
    def getSpecies(self):
        hc = []
        fuelch4 = []
        for species in self.species:
            if species =='H2O' and self._phase == 'Liquid' or self._phase == 'liquid':
                fuelch4.append('H2O(L)')
            elif species in ['H2', 'CO2', 'H2O', 'CO', 'N2', 'H2O(L)', 'CH4', 'O2', 'NH3']: 
                fuelch4.append(species)
            else:                    
                hc.append(species)
        return fuelch4, hc
    
    def getMW(self): # data in HC
        return np.dot(self._MWi, self._Xi)
    
    def calVAP(self): # only for HC
        '''
        Method that returns the enthalpy of vaporization of hydrocarbons.
        - The n oefficient and Tc were taken from "Yaws, Carl. Thermophysical Properties of Chemicals and Hydrocarbons. 2nd Ed., 2014".
        - The HVAP_B coefficient is calculated with the HVAP and the boiling T as: HVAP_B = Hvap_boil/(1 - Tboil/Tc)**n. Hvap_boil was 
        obtained from the same reference.
        - The sensible heat when T < Tmin is accounted. Cp°liquid(cpl) was taken from 
        '''

        vap = _VapCalc(self._T, self.__HVAP_Tmin, self.__HVAP_Tmax, self.__HVAP_B, self.__HVAP_Tc, self.__HVAP_n, self.__cpl)
        return vap 

    def thermoArr(self): 
        self._thermArr = _thermoArrStat(self._T)

    def calcThermoHC(self): 

        VAP = _phaseVap(self._phaseID, self._T, self.__HVAP_Tmin, self.__HVAP_Tmax, self.__HVAP_B, self.__HVAP_Tc, self.__HVAP_n, self.__cpl, self.__NHC)
        self.__HVAP = VAP[0]
        self.__SVAP = VAP[1]

        self.__therm_sep = _mtCalcStatic(self.__m, self.__c, self._thermArr, self.CNumberi) 
        
        self.__HC_cp = self.__therm_sep[0]
        self.__HC_h = self.__therm_sep[1] - self.__HVAP 
        self.__HC_s = self.__therm_sep[2] - self.__SVAP
        self.__HC_g = self.__HC_h - self._T*self.__HC_s   
    
    def calcThermoSep(self):
        
        self.cp_i = np.concatenate((self.__FuelCH4_Th.cp_i, self.__HC_cp)) 
        self.h_i = np.concatenate((self.__FuelCH4_Th.h_i, self.__HC_h)) 
        self.s_i = np.concatenate((self.__FuelCH4_Th.s_i, self.__HC_s)) 
        self.g_i = np.concatenate((self.__FuelCH4_Th.g_i, self.__HC_g)) 
    
    def calcThermoMix(self):
        
        thermMix = th._ThermoMix(self._Xi, self.cp_i, self.h_i, self.s_i, self._P)
        self.cp = thermMix[0] 
        self.h = thermMix[1] 
        self.s_T = thermMix[2] 
        self.s = self.s_T - self.R*thermMix[3] 
        self.g = self.h - self._T*self.s

    @property
    def LHVmix(self): # data in HC
        return np.dot(self.__DATA['lhv'], self._Xi)
        
    # LHV of a given specie
    def LHVi(self, specie):
        return  self.__DATA['lhv'][specie]
    
    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, ph):
        self._phase = ph
        if self.__NHC > 0:
            self.calcThermoHC()
        else: # temporary
            self.__HC_cp, self.__HC_h, self.__HC_s, self.__HC_g = np.array([]), np.array([]), np.array([]),np.array([])
        self.calcThermoSep()
        self.calcThermoMix()
      
    ## Poperties
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, t):
        self._T = t 
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def Xi(self):
        return self._Xi
    
    @Xi.setter
    def Xi(self, x):
        self._Xi = x
        self.calcThermoMix()
        
    @property
    def P(self):
        return self._P
        
    @P.setter
    def P(self, p):
        self._P = p     
        self.__FuelCH4_Th.P = self._P
        self.calcThermoSep()
        self.calcThermoMix()
    
    @property
    def TX(self):
        return self._T, self._Xi
        
    @TX.setter
    def TX(self, tx):
        t, x = tx
        
        self._T = t 
        self._Xi = x
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def TcX(self):
        return self._T-273.15, self._Xi
    
    @TcX.setter
    def TcX(self, tx):
        t,x = tx
            
        self._T = t + 273.15
        self._Xi = x
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def Tc(self):
        return self._T - 273.15
        
    @Tc.setter
    def Tc(self, t):
        self._T = t + 273.15
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def PTX(self):
        return self._P, self._T, self._Xi
        
    @PTX.setter
    def PTX(self, ptx):
        p,t,x = ptx
        
        self._T = t 
        self._P = p
        self._Xi = x
        self.__FuelCH4_Th.P = self._P
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def PTcX(self):
        return self._P, self._T - 273.15, self._Xi
        
    @PTcX.setter
    def PTcX(self, ptx):
        p,t,x = ptx
        
        self._T = t + 273.15
        self._P = p
        self._Xi = x
        self.__FuelCH4_Th.P = self._P
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()

''' 
    @classmethod
    def getPsat(cls, T, species):   
        # returns the saturation pressure in bar. T in Kelvin
        A = cls.df[1]['A'][species]
        B = cls.df[1]['B'][species]
        C = cls.df[1]['C'][species]
        T_C = T - 273.15 # in celcius
        ## hypothetical vapor pressure for non-condensables from "Sun, Chenghao PhD Thesis 2017: Direct syngas-to-fuel:integration of FT
                                                                 synthesis and hydrocraking in micro-structured reactors, page 41"
        if species in ['H2', 'CO', 'H2O', 'N2', 'CH4', 'C2H6', 'C3H8', 'C4H10', 'C5H12']:
            P_sat = A + B*np.exp(-T/C)
        else:
            ## from "Yaws Handbook of vapor pressure Antoine Coeff" 2nd Ed., 2015
            P_sat = (10**(A-B/(T_C+C)))*0.00133322
        return P_sat
    
    @classmethod
    def Psat(cls, T, speciesList):
        Psat = np.zeros(len(speciesList))
        for i, var in enumerate(speciesList):
            Psat[i] = cls.getPsat(T, var)
        return Psat
    
    @classmethod
    def Pbubble(cls, T, Xi, speciesList):
        Pbubble = 0
        for i, var in enumerate(speciesList):
            Pbubble += np.where(Xi[i] > 1e-6,  Xi[i]*cls.getPsat(T, var), 0)
        return Pbubble
    
    @classmethod
    def Pdew(cls, T, Xi, speciesList):
        Pdew = 0
        for i, var in enumerate(speciesList):
            Pdew +=np.where(Xi[i] > 1e-6, Xi[i]/cls.getPsat(T, var), 0)      
        return 1/Pdew
''' 
    
class ThermoExergy:   

    file_path = files("celeste.data").joinpath("Chem_exergies.csv").as_posix() 
    df = pd.read_csv(file_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    df = df.set_index("Name")
    
    Ps = 0.03169 #0.03137185#
    T0 = 298.15
    P0 = 1 # bar
    R =  8.314462618e-3 
    
    def __init__(self, Medium, Xi=None, T=None, P=None, phase='Gas'): 
        self.Medium = Medium         
        self.Xi = np.where(Xi>=1e-6, Xi, 1e-12)
        self.T = T
        self.P = P
        self.phase = phase
        
        ## thermo at PTXi
        self.Medium.phase = self.phase # has to be checked with the FT
        self.Medium.PTX = self.P, self.T, self.Xi
        self.__h_i = self.Medium.h_i
        self.__s_i = self.Medium.s_i # s for each specie without P effect
        self.__th_T0 = Thermo(species=self.Medium.species, T=self.T0, Xi=self.Xi, P=self.P0, phase=self.phase)
        self.__h0_i = self.__th_T0.h_i
        self.__s0_i = self.__th_T0.s_i # s for each specie without P effect @ standard conditions
        ### thermo of liquid water at T0
        self.__th_H2OL_T0 = th.Thermo(species=['H2O(L)'], T=self.T0, Xi=np.array([1]), P=self.P0)
        self.__h0__H2OL = self.__th_H2OL_T0.h
        self.__s0__H2OL = self.__th_H2OL_T0.s 
            
        self.__x_00_H2O = self.Ps/self.P0 # Raoult's law
        
        if 'H2O' in self.Medium.species: # only for steam
            self.__H2O = self.Medium.species.index('H2O')    
        else:
            self.__H2O = None   

        if 'H2O(L)' in self.Medium.species: # only for liquid H2O
            self.__H2OL = self.Medium.species.index('H2O(L)') 
        else:
            self.__H2OL = None   
        # standard chemical exergy [kJ/mol] at 298.15 and 100 kPa from Baehr, 2012, 'Thermodynamik' --> changed to another source
        # For alkanes from C5-C50 it is calculated with the h°f(l) and s°f(l)
        self.__stEx = np.array(self.df['Standard Chem Exergy kJ/mol'][self.Medium.species], dtype=float)
        self.__stEx_H2OL = np.array(self.df['Standard Chem Exergy kJ/mol']['H2O(L)'], dtype=float)
        
    @property
    def Ex(self):
        return self.streamPhysExergy() + self.streamChemExergy() 
    
        
    def streamPhysExergy(self):       
        if self.__H2O is not None: # steam
            # For gas mixtures with steam, there is steam condensation at the dead state conditions
            # Therefore need to normalise other gases to new water composition
            if self.Xi[self.__H2O] <= self.__x_00_H2O: # When water fraction is less than saturation, no condensation
                # Calculate enthalpy and entropy change of each species
                dH_T_i = self.__h_i - self.__h0_i
                dS_TP_i = self.__s_i - self.__s0_i - self.R*np.log(self.P)   
            else: # When water fraction is more than saturation, condensation will occur
                # Calculate gas phase composition at dead state
                if self.Xi[self.__H2O] == 1:
                    x_0 = self.Xi
                else:
                    x_0 = self.Xi*(1-self.__x_00_H2O)/(1 - self.Xi[self.__H2O])
                    x_0[self.__H2O] = self.__x_00_H2O   
                
                n_V = (1 - self.Xi[self.__H2O])/(1 - self.__x_00_H2O) # Ratio of stream that is in gas/vapour phase at dead state 
                n_L = 1 - n_V # Ratio of stream that condenses at dead state conditions
                n_H2O = n_L + n_V*x_0[self.__H2O]  
                # Calculate enthalpy and entropy change of each species  
                dH_T_i = self.__h_i - self.__h0_i 
                dS_TP_i = self.__s_i - self.__s0_i - self.R*np.log(self.Xi*self.P/x_0, where=(x_0>=1e-6), out=np.zeros_like(self.Xi))
                dH_T_i[self.__H2O]  = (n_H2O*self.__h_i[self.__H2O] - n_L*self.__h0__H2OL - x_0[self.__H2O]*n_V*self.__h0_i[self.__H2O])/n_H2O
                dS_TP_i[self.__H2O] =  (n_H2O*(self.__s_i[self.__H2O] - self.R*np.log(n_H2O*self.P)) - n_L*self.__s0__H2OL - 
                                      x_0[self.__H2O]*n_V*(self.__s0_i[self.__H2O] - self.R*np.log(x_0[self.__H2O])) )/n_H2O
        else:
            # For gas mixtures that do not contain steam, there is no condensation (need to modify for gasses that do condense)
            # Calculate enthalpy and entropy change of each species
            dS_TP_i = self.__s_i - self.__s0_i  - self.R*np.log(self.P)
            dH_T_i = self.__h_i - self.__h0_i
            if self.__H2OL: # entropy not affected by pressure
                dS_TP_i[self.__H2OL] = self.__s_i[self.__H2OL] - self.__s0__H2OL

        # Calculate Physical Exergy
        dH_T = np.dot(self.Xi, dH_T_i)
        dS_TP = np.dot(self.Xi, dS_TP_i)
        Ex_ph = dH_T - self.T0*dS_TP
        return Ex_ph
    
    def streamChemExergy(self):
        if self.__H2O is not None:
            if self.Xi[self.__H2O] <= self.__x_00_H2O: 
                x_0 = self.Xi
                # Calculate Chemical Exergy
                Ex_ch_i = self.__stEx + self.R*self.T0*np.log(x_0, where=(x_0>=1e-6), out=np.zeros_like(x_0))   
            else:          
                if self.Xi[self.__H2O] == 1:
                    x_0 = self.Xi
                else:
                    x_0 = self.Xi*(1-self.__x_00_H2O)/(1 - self.Xi[self.__H2O])
                    x_0[self.__H2O] = self.__x_00_H2O  
                n_V = (1 - self.Xi[self.__H2O])/(1 - self.__x_00_H2O) # Ratio of stream that is in gas/vapour phase at dead state 
                n_L = 1 - n_V # Ratio of stream that condenses at dead state conditions
                n_H2O = n_L + n_V*x_0[self.__H2O] 
                # Calculate Chemical Exergy
                Ex_ch_i = self.__stEx + self.R*self.T0*np.log(x_0, where=(x_0>=1e-6), out=np.zeros_like(x_0))
                Ex_ch_i[self.__H2O] = (x_0[self.__H2O]*n_V*(self.__stEx[self.__H2O] + self.R*self.T0*np.log(x_0[self.__H2O])) + n_L*self.__stEx_H2OL)/n_H2O
       
        else:
            Ex_ch_i = self.__stEx + self.R*self.T0*np.log(self.Xi, where=(self.Xi>=1e-6), out=np.zeros_like(self.Xi)) 
            if self.__H2OL is not None:
                Ex_ch_i[self.__H2OL] = self.__stEx[self.__H2OL]
            
        Ex_ch = np.dot(self.Xi, Ex_ch_i) 
        return Ex_ch     
