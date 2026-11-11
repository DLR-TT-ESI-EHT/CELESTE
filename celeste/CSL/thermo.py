import numpy as np
import pandas as pd
import numba as nb
import json
#from scipy import constants
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files
import CoolProp.CoolProp as CP


'''
1. Thermo: calculation of MW, cp, h, s, g of a single component or stream for NASA components
2. ThermoDum: dummy thermo object. Returns empty arrays for cp, h, s and g
3. Thermo_CP: CoolProp based calculation of thermo properties for single component or mixtures using HEOS, PR or SRK
'''

@nb.njit
def _ThermoMix(Xi, cp_i, h_i, s_i, P):
    cp = 0.
    h = 0.
    s_T = 0.
    s_P = 0.

    for i,v in enumerate(Xi):
        if Xi[i] >=1e-6:
            cp += Xi[i]*cp_i[i]
            h += Xi[i]*h_i[i]
            s_T += Xi[i]*s_i[i]
            s_P += Xi[i]*np.log(Xi[i]*P)

    return np.array([cp, h, s_T, s_P])

@nb.njit
def _thermoArrStatic(t):
    '''shape nCoeffs, nThermoProps(3)'''

    T2 = t*t
    T3 = T2*t
    T4 = T3*t
    T5 = T4*t
    T_2 = 1/T2
    T_1 = 1/t
    ln_T = np.log(t)

    thermArr = np.array([
        [T_2,      T_1,   1.,       t,  T2,         T3,        T4    ,  0., 0.],
        [-T_1,     ln_T,  t,  0.5*T2,   (1/3)*T3,   0.25*T4,   0.2*T5,  1., 0.],
        [-0.5*T_2, -T_1,  ln_T,     t,   0.5*T2,    (1/3)*T3,  0.25*T4, 0., 1.]
    ]) 
    return thermArr

@nb.njit
def _calcThermoSepStatic(tArr, a, idxBig):
    I,J,_ = a.shape
    thermo_sep = np.zeros((3,I))     
    for i in nb.prange(I):
        for j in range(J):
            for k in range(3):
                for l in range(9):
                    thermo_sep[k,i] += tArr[k,l]*idxBig[i,j]*a[i,j,l]
    return thermo_sep

def _calcThermoSepIDX(tArr, a, idxBig):
    therSep = tArr.dot(a[idxBig, :].T)
    return therSep

class Thermo:
    with open(files("celeste.data").joinpath("coeffsData.json").as_posix(), "r") as f:
        dataNASA = json.load(f)
    df = pd.json_normalize(dataNASA).set_index('speciesName')

    s_lhv = pd.Series({'H2': 241.809, 'CH4': 802.262, 'CO2': 0, 'CO': 282.98, 'H2O':0, 'N2':0, 'H2O(L)': 0, 'O2':0, 'CH3OH':637.6358+28.38906, 'CH3OH(L)':637.6358, #Add latent heat to gas MeOH (28.38906)
                       'NH3': 316.836, 'C3H8':2044.2, 'C(gr)': 393.5})# https://cantera.org/examples/jupyter/thermo/heating_value.ipynb.html
    R = 8.314462618e-3
    
    def __init__(self, species, T:float, Xi, P:float=1., T_unit:str='K'):
        
        self.species = species # nSpecies
        self.Ns = len(self.species)  
        
        if T_unit in ['C', 'c']:
            T = T + 273.15
        
        self._T = T #np.array(T,dtype=float) 
        self._Xi = Xi # nSpecies
        self._P = P
        
        self.__DATA = self.df.loc[self.species]
        self.__DATA['lhv'] = self.s_lhv[self.species]
        
        self.maxTint = self.__DATA['nTIntervals'].max()
        self._MWi = np.array(self.__DATA['MolecularWeight'], dtype=float)
        self._idxBig = np.zeros((self.Ns, self.maxTint), dtype=bool) # nT, nSpecies, nTempIntervals

        if 'H2O(L)' in self.species: 
            self.__H2OL = self.species.index('H2O(L)') 
        else:
            self.__H2OL = None   
        
        self.getCoeffs()
        self.updateBoolArr()
        self.thermoArr()      
        self.calcThermoSep()
        self.calcThermoMix()
    
    ## Methods    
    def getMW(self):
        return np.dot(self._MWi, self._Xi)
        
    def getCoeffs(self):
        self.__T_min = np.zeros((self.Ns, self.maxTint)) # nSpecies, nTempIntervals
        self.__T_max = np.zeros((self.Ns, self.maxTint)) # nSpecies, nTempIntervals
        self.__a = np.zeros((self.Ns, self.maxTint, 9)) # nSpecies, nTempIntervals, nPolynomialCoeffs(7)
        
        for i,v in enumerate(self.species):
            nTInt_i = self.__DATA['nTIntervals'][v]
            TRange = self.__DATA['TemperatureRange'][v]
            
            self.__T_min[i,  :nTInt_i] = [TRange[j]['T_min'] for j in range(nTInt_i)]
            self.__T_max[i,  :nTInt_i] = [TRange[j]['T_max'] for j in range(nTInt_i)]

            self.__a[i, :nTInt_i, :7] = [TRange[j]['a'] for j in range(nTInt_i)]
            self.__a[i, :nTInt_i, 7:] = [TRange[j]['b'] for j in range(nTInt_i)]
            
    def updateBoolArr(self):
        self._idxBig = np.logical_and(  self._T > self.__T_min, self._T <= self.__T_max )

    def thermoArr(self):
        self._thermArr = _thermoArrStatic(self._T)
        return self._thermArr

    def calcThermoSep(self):
        '''nSpecies, [cp, h, s]'''
        
        #self.__therm_sep = self.R*self.__calcThermoSepIDX(self._thermArr, self.__a, self._idxBig)   #np.einsum('jkl, jk, ml->mj', self.__a, self._idxBig, self._thermArr)
        self.__therm_sep = self.R*_calcThermoSepStatic(self._thermArr, self.__a, self._idxBig)

        self.cp_i = self.__therm_sep[0]
        self.h_i = self.__therm_sep[1]
        self.s_i = self.__therm_sep[2]  
        self.g_i = self.h_i - self._T*self.s_i
        
    def calcThermoMix(self):

        self.__therm_mix = _ThermoMix(self._Xi, self.__therm_sep[0], self.__therm_sep[1], self.__therm_sep[2], self._P) #np.dot(self.__therm_sep, self._Xi)
        self.cp = self.__therm_mix[0] # np.einsum('mi,mi->m', self.cp_i, self.Xi) # therm_mix[0]
        self.h = self.__therm_mix[1] # np.einsum('mi,mi->m', self.h_i, self.Xi) # therm_mix[1]
        self.s_T = self.__therm_mix[2] # np.einsum('mi,mi->m', self.s_i, self.Xi) # therm_mix[2]
        
        if self.__H2OL:
            self.s = self.s_T - (self.R*self.__therm_mix[3] - self.R*self.Xi[self.__H2OL]*np.log(self._Xi[self.__H2OL]*self._P,
                                                              out=np.zeros_like(self._Xi[self.__H2OL], dtype=np.float64), where=(self.Xi[self.__H2OL]>=1e-6)) ) 
        else:
            self.s = self.s_T - self.R*self.__therm_mix[3]

        self.g = self.h - self._T*self.s
    
    # Calculate LHV of mixture
    @property
    def LHVmix(self):
        return np.dot(self.__DATA['lhv'], self.Xi)
        
    # LHV of a given specie
    @property
    def LHVi(self):
        return self.__DATA['lhv'][self.species]
        
    ## Poperties
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, t):
        self._T = t # np.array(t, dtype=float) 
        self.updateBoolArr()
        self.thermoArr()
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
    def Yi(self):
        return self._Xi*self._MWi/self.getMW()

    @Yi.setter
    def Yi(self, y):
        self.Xi = (y/self._MWi)/np.sum(y/self._MWi)

    @property
    def P(self):
        return self._P
        
    @P.setter
    def P(self, p):
        self._P = p
        self.calcThermoSep()
        self.calcThermoMix()
    
    @property
    def TX(self):
        return self._T, self._Xi
        
    @TX.setter
    def TX(self, tx):
        t, x = tx
        
        self._T = t # np.array(t, dtype=float)
        self._Xi = x
        self.updateBoolArr()
        self.thermoArr()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def TcX(self):
        return self._T-273.15, self._Xi
    
    @TcX.setter
    def TcX(self, tx):
        t,x = tx
        
        self._T = t+273.15 #+ np.array(t, dtype=float) + 273.15
        self._Xi = x
        self.updateBoolArr()
        self.thermoArr()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def Tc(self):
        return self._T - 273.15
        
    @Tc.setter
    def Tc(self, t):
        self._T = t+273.15 # np.array(t, dtype=float) + 273.15
        self.updateBoolArr()
        self.thermoArr()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def PTX(self):
        return self._P, self._T, self._Xi
        
    @PTX.setter
    def PTX(self, ptx):
        p,t,x = ptx
        
        self._T = t # np.array(t, dtype=float)
        self._P = p
        self._Xi = x
        self.updateBoolArr()
        self.thermoArr()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def PTcX(self):
        return self._P, self._T - 273.15, self._Xi
        
    @PTcX.setter
    def PTcX(self, ptx):
        p,t,x = ptx

        self._T = t+273.15 #np.array(t, dtype=float) + 273.15
        self._P = p
        self._Xi = x
        self.updateBoolArr()
        self.thermoArr()
        self.calcThermoSep()
        self.calcThermoMix()

    def __copy__(self): # makes completely new object when copy is called
        return Thermo(self.species, self._T, self._Xi, self._P)


class ThermoDum():
    def __init__(self, T):
        self._T = T
        self.P = None
        self.cp_i = np.array([])
        self.h_i = np.array([])
        self.s_i = np.array([])
        self.g_i = np.array([])
        
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, T):
        self.__dict__['T'] = T
        self._T = T
        self.cp_i = np.array([])
        self.h_i = np.array([])
        self.s_i = np.array([])
        self.g_i = np.array([])
        
class Thermo_CP:
    '''
    CoolProp Based Thermo Object 

    Inputs:
        species:    array of species; np.array[len=nSpecies, dtype=str]
        T:          Temperature     ; float    
        Xi:         mole fractions  ; np.array[len=nSpecies, dtype=float]
        P:          Pressure        ; float
        EOS:        Equation of state;str {'HEOS', 'PR', 'SRK'}; default: 'HEOS'
        phase:      fluid phase     ; str {'gas', 'liquid'} | int {cp.iphase_}; default; 'gas'

    Attrs:
        T:  Temperature, K      ; float
        P:  Pressure, bar       ; float
        Xi: Mole Fraction, -    ; np.Array[len=nSpecies, dtype=float]
        h:  Enthalpy, kJ/mol    ; float                 ; Difference from Standard Conditions
        s:  Entropy, kJ/mol/K   ; float                 ; Difference from Standard Conditions
        g:  Gibbs Energy, kJ/mol; float                 ; H - TS
        cp: Specific Heat Cap.  ; float
        Mixed: TP, TPX          ; Tuple(T, P), Tuple(T, P, Xi)

    Private Attrs:
        _Thermo_CP__AS: CoolProp AbstractState with current conditions. Can use the Thermo_CP object to access methods in this object.
        _Thermo_CP__REF:CoolProp AbstractState with reference conditions. Used to calculate H and S difference
    '''
    #Creating a phase dictionary to convert phase index into a phase string
    phase_dict = {
        CP.iphase_liquid: "liquid",
        CP.iphase_supercritical_liquid: "supercritical_liquid",
        CP.iphase_supercritical_gas: "supercritical_gas",
        CP.iphase_gas: "gas",
        CP.iphase_supercritical: "supercritical",
        CP.iphase_twophase: "twophase",
        CP.iphase_unknown: "unknown"
    }

    def __init__(self, species, T, Xi, P, EOS='HEOS', phase=None): 
        #with EOS as HEOS, true phase at the state conditions can be retrieved but not for PR. 
        #HEOS doesnt work for mixtures due to binary interaction error.
        self.species = species
        self._T = T
        self._P = P
        self._Xi = Xi
        self.EOS = EOS
        self._phase = phase

        if phase in ['gas', 'Gas']:
            self.__CP_phase = CP.iphase_gas
        elif phase in ['liquid', 'Liquid']:
            self.__CP_phase = CP.iphase_liquid
        elif phase in ['supercritical_gas', 'Supercritical_gas']:
            self.__CP_phase = CP.iphase_supercritical_gas
        elif phase in ['supercritical_liquid', 'Supercritical_liquid']:
            self.__CP_phase = CP.iphase_supercritical_liquid
        elif phase in ['supercritical', 'Supercritical']:
            self.__CP_phase = CP.iphase_supercritical
        elif phase in ['twophase', 'Twophase']:
            self.__CP_phase = CP.iphase_twophase
        else:
            self.__CP_phase = phase
            #raise ValueError(f"Invalid phase '{phase}'. Expected 'gas', 'liquid', 'supercritical_gas', 'supercritical_liquid', 'supercritical', or 'twophase'.")

        self.__REF = CP.AbstractState(EOS, '&'.join(species))
        self.__REF.set_mole_fractions(Xi)
        #self.__REF.specify_phase(self.__CP_phase) #probably not needed
        self.__update_ref()

        self.__REF_Cel = Thermo(species, 298.15, Xi, 1.)
        
        self.__AS = CP.AbstractState(EOS, '&'.join(species))
        self.__AS.set_mole_fractions(Xi)

        self.__AS.specify_phase(self.__CP_phase)
                
        try:
            # Attempt to specify the phase
            self.__AS.specify_phase(self.__CP_phase)
            # Attempt to update the state with the specified P and T
            self.__update_AS()
        except ValueError as e:
            # Print the warning only if there's an actual ValueError
            print('Warning: Invalid phase input at the given Temperature and Pressure')
            print(f"Error Details: {e}")
            return  # Exit the method to avoid further processing

    def __update_ref(self): # will be called loads of times
        self.__REF.update(CP.PT_INPUTS, 101325, 298.15)

    def __update_AS(self):
        self.__AS.update(CP.PT_INPUTS, self._P * 1e5, self._T)
    
        

    def getMW(self):
        return self.__AS.molar_mass() * 1e3  # g/mol

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, t):
        self._T = t
        self.__update_AS()

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, p):
        self._P = p
        self.__update_AS()

    @property
    def TP(self):
        return self._T, self._P

    @TP.setter
    def TP(self, tp):
        self._T, self._P = tp
        self.__update_AS()

    @property
    def Xi(self):
        return self._Xi

    @Xi.setter
    def Xi(self, xi):
        self._Xi = xi
        self.__AS.set_mole_fractions(xi)
        self.__update_AS()
        self.__REF.set_mole_fractions(xi)
        self.__update_ref()
        self.__REF_Cel.Xi = xi

    @property
    def PTX(self):
        return self._P, self._T, self._Xi

    @PTX.setter
    def PTX(self, ptx):
        self._P, self._T = ptx[:2]
        self.Xi = ptx[2]

    @property
    def h(self):
        return self.__REF_Cel.h + (self.__AS.hmolar() - self.__REF.hmolar()) * 1e-3

    @property
    def s(self):
        return self.__REF_Cel.s + (self.__AS.smolar() - self.__REF.smolar()) * 1e-3

    @property
    def g(self):
        return self.h - self.T * self.s

    @property
    def cp(self):
        return self.__AS.cpmolar()

    @property
    def Yi(self):
        return self.__AS.get_mass_fractions()

    @Yi.setter
    def Yi(self, y):
        self.__AS.set_mass_fractions(y)
        self.Xi = np.array(self.__AS.get_mole_fractions())

    @property
    def phase(self):
        
        return self._phase

    @phase.setter
    def phase(self, ph):
        if ph in ['gas', 'Gas']:
            self.__CP_phase = CP.iphase_gas
        elif ph in ['liquid', 'Liquid']:
            self.__CP_phase = CP.iphase_liquid
        elif ph in ['supercritical_gas', 'Supercritical_gas']:
            self.__CP_phase = CP.iphase_supercritical_gas
        elif ph in ['supercritical_liquid', 'Supercritical_liquid']:
            self.__CP_phase = CP.iphase_supercritical_liquid
        elif ph in ['supercritical', 'Supercritical']:
            self.__CP_phase = CP.iphase_supercritical
        elif ph in ['twophase', 'Twophase']:
            self.__CP_phase = CP.iphase_twophase
        else:
            self.__CP_phase = ph
        
        self.__AS.specify_phase(self.__CP_phase)
    
    def __copy__(self): # makes completely new object when copy is called
        return Thermo_CP(self.species, self._T, self._Xi, self._P, self.EOS, self._phase)
