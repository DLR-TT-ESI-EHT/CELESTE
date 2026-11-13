import numpy as np
from numpy.typing import NDArray
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

    - pure-species low-pressure viscosity: Chung et al. (1984/1988) correlation, using
      each species' critical temperature Tc, critical molar volume Vc, acentric factor w
      and dipole moment mu. Same data and constants as Modelica.Media.IdealGases
      (FluidData.mo); like the MSL, the association correction kappa is NOT applied (kappa = 0).
    - pure-species thermal conductivity: Eucken (method 1, MSL default) / modified-Eucken
      (method 2) correlation built on top of the viscosity above and the species' own NASA cp(T)
    - mixture viscosity: Wilke's method (Modelica's gasMixtureViscosity).
    - mixture thermal conductivity: Mason & Saxena's method as implemented in Modelica's
      lowPressureThermalConductivity (epsilon = 1.0; interaction term built from
      Gamma = 210*(Tc*M^3/Pc^4)^(1/6) and the reduced temperature, NOT from the Wilke phi_ij).
 
Species that are not covered by the low-pressure-gas theory above (the liquid pseudo-species 'H2O(L)'
and 'CH3OH(L)' used elsewhere in this package) fall back to simple literature
viscosity/conductivity correlations instead.
'''

R_UNIV = 8.314462618  # J/(mol.K)

# Critical/polar data, identical to Modelica.Media.IdealGases.Common.FluidData.
# Tc: critical temperature [K]; Pc: critical pressure [Pa] (needed by the Mason-Saxena
# mixture conductivity); Vc: critical molar volume [cm3/mol]; w: acentric factor [-];
# mu: dipole moment [Debye]; kappa: Chung's association correction. The MSL never passes
# kappa to dynamicViscosityLowPressure (default k = 0), so it is 0 here for every species.
# For a physically better (but non-MSL) water/methanol viscosity use kappa=0.076 / 0.215.

FLUID_DATA = {
    'H2':    dict(Tc=33.25,   Pc=12.97e5,  Vc=65.00,  w=-0.216, mu=0.0, kappa=0.0),
    'CH4':   dict(Tc=190.56,  Pc=45.99e5,  Vc=98.60,  w=0.011,  mu=0.0, kappa=0.0),
    'CO2':   dict(Tc=304.12,  Pc=73.74e5,  Vc=94.07,  w=0.225,  mu=0.0, kappa=0.0),
    'CO':    dict(Tc=132.85,  Pc=34.94e5,  Vc=93.10,  w=0.045,  mu=0.1, kappa=0.0),
    'H2O':   dict(Tc=647.096, Pc=220.64e5, Vc=55.95,  w=0.344,  mu=1.8, kappa=0.0),
    'N2':    dict(Tc=126.20,  Pc=33.98e5,  Vc=90.10,  w=0.037,  mu=0.0, kappa=0.0),
    'O2':    dict(Tc=154.58,  Pc=50.43e5,  Vc=73.37,  w=0.022,  mu=0.0, kappa=0.0),
    'NH3':   dict(Tc=405.40,  Pc=113.53e5, Vc=72.47,  w=0.257,  mu=1.5, kappa=0.0),
    'CH3OH': dict(Tc=512.64,  Pc=80.97e5,  Vc=118.00, w=0.565,  mu=1.7, kappa=0.0),
    'C3H8':  dict(Tc=369.83,  Pc=42.48e5,  Vc=200.00, w=0.152,  mu=0.0, kappa=0.0),
}

# Species handled through liquid-state engineering correlations instead of the
# low-pressure-gas theory above (see _waterLiquid*/_methanolLiquid* below).
_LIQUID_TRANSPORT_SPECIES = ('H2O(L)', 'CH3OH(L)')


@nb.njit
def _omegaV(Tstar):
    '''Neufeld collision integral, Omega_v(T*), used by the Chung viscosity correlation.'''
    return (1.16145*Tstar**-0.14874 + 0.52487*np.exp(-0.77320*Tstar) + 2.16178*np.exp(-2.43787*Tstar))


@nb.njit
def _chungViscosityArr(T, Mi, Tci, Vci, wi, mui, kappai):
    '''Pure-species low-pressure dynamic viscosity / Pa.s (Chung et al. 1984/1988).

    T: gas temperature, K. Mi: molar mass, g/mol. Tci: critical temperature, K.
    Vci: critical molar volume, cm3/mol. wi: acentric factor. mui: dipole moment, Debye.
    kappai: empirical association correction.
    '''
    n = Mi.shape[0]
    eta = np.zeros(n)
    for i in range(n):
        mur = 131.3*mui[i]/np.sqrt(Vci[i]*Tci[i])
        Fc = 1. - 0.2756*wi[i] + 0.059035*mur**4 + kappai[i]
        Tstar = 1.2593*T/Tci[i]
        Om = _omegaV(Tstar)
        eta_muP = 40.785*Fc*np.sqrt(Mi[i]*T)/(Vci[i]**(2./3.)*Om)  # micropoise
        eta[i] = eta_muP*1e-7  # Pa.s
    return eta


@nb.njit
def _euckenConductivityArr(eta_i, cp_i, R_i, method):
    '''Pure-species thermal conductivity, W/(m.K), from the Eucken (method=1) or
    modified-Eucken (method=2) correlation. eta_i: viscosity, Pa.s. cp_i, R_i: mass-specific
    cp and gas constant, J/(kg.K).
    '''
    n = eta_i.shape[0]
    k = np.zeros(n)
    for i in range(n):
        if method == 1:
            k[i] = eta_i[i]*(cp_i[i] - R_i[i] + 2.25*R_i[i])
        else:
            cv = cp_i[i] - R_i[i]
            k[i] = eta_i[i]*(1.32*cv + 1.77*R_i[i])
    return k


@nb.njit
def _wilkePhi(eta_i, Mi):
    '''Wilke interaction parameters phi_ij (also used as the Mason-Saxena parameters
    for mixture thermal conductivity).'''
    n = eta_i.shape[0]
    phi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            phi[i, j] = (1. + np.sqrt(eta_i[i]/eta_i[j])*(Mi[j]/Mi[i])**0.25)**2 / np.sqrt(8.*(1. + Mi[i]/Mi[j]))
    return phi


@nb.njit
def _wilkeMixViscosity(Xi, eta_i, Mi):
    '''Mixture viscosity, Pa.s, from Wilke's method.'''
    n = Xi.shape[0]
    phi = _wilkePhi(eta_i, Mi)
    etam = 0.
    for i in range(n):
        if Xi[i] >= 1e-6:
            denom = 0.
            for j in range(n):
                if Xi[j] >= 1e-6:
                    denom += Xi[j]*phi[i, j]
            etam += Xi[i]*eta_i[i]/denom
    return etam


@nb.njit
def _masonSaxenaMixConductivity(Xi, k_i, Tci, Pci, Mi, T):
    '''Mixture thermal conductivity, W/(m.K): Mason & Saxena as in Modelica's
    lowPressureThermalConductivity (epsilon = 1.0).
 
    Xi: mole fractions. k_i: pure-species conductivities, W/(m.K). Tci: critical
    temperatures, K. Pci: critical pressures, Pa. Mi: molar masses (any unit, only
    ratios enter). T: gas temperature, K.
    '''
    n = Xi.shape[0]
    gam = np.zeros(n)
    f = np.zeros(n)
    for i in range(n):
        gam[i] = 210.*(Tci[i]*Mi[i]**3/Pci[i]**4)**(1./6.)
        Tr = T/Tci[i]
        f[i] = np.exp(0.0464*Tr) - np.exp(-0.2412*Tr)
    km = 0.
    for i in range(n):
        denom = 0.
        for j in range(n):
            ratio = gam[j]*f[i]/(gam[i]*f[j])  # translational-conductivity ratio lambda_i/lambda_j
            A = (1. + np.sqrt(ratio)*(Mi[i]/Mi[j])**0.25)**2/np.sqrt(8.*(1. + Mi[i]/Mi[j]))
            denom += Xi[j]*A
        km += Xi[i]*k_i[i]/denom
    return km


def _waterLiquidViscosity(T):
    '''Liquid water dynamic viscosity, Pa.s. Engineering correlation, valid roughly
    273-600 K (same accuracy class as the Yaws-handbook density fits already used
    elsewhere in this package for H2O(L)).'''
    return 2.414e-5*10**(247.8/(T - 140.))


def _waterLiquidConductivity(T):
    '''Liquid water thermal conductivity, W/(m.K). Quadratic engineering fit,
    valid roughly 273-600 K.'''
    return -0.5752 + 6.397e-3*T - 8.151e-6*T**2


def _methanolLiquidViscosity(T):
    '''Liquid methanol dynamic viscosity, Pa.s. Andrade-type fit to literature data,
    valid roughly 250-350 K.'''
    return np.exp(-11.772 + 1272.7/T)


def _methanolLiquidConductivity(T):
    '''Liquid methanol thermal conductivity, W/(m.K). Linear fit to literature data,
    valid roughly 260-330 K.'''
    return 0.202 - 4.0e-4*(T - 293.15)

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
    
    # 1: standard Eucken correlation (Modelica's default)
    # 2: modified Eucken correlation (generally more accurate for polyatomic species, Poling et al.). Can be overridden per-instance, e.g. `myMedium.condMethod = 1`.
    condMethod = 2
    
    def __init__(self, species, T:float, Xi, P:float=1., T_unit:str='K'):
        
        self.species = species # nSpecies
        self.Ns = len(self.species)  
        
        if T_unit in ['C', 'c']:
            T = T + 273.15
        
        self._T: float = T #np.array(T,dtype=float) 
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
        self.setupTransport()
    
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
        
    def setupTransport(self):
        '''Build the per-species transport-property tables (critical/polar data for the
        gas-phase species, flags for the liquid pseudo-species). Called once at
        construction, since self.species never changes over the life of the object.'''

        n = self.Ns
        self._Tc_i = np.ones(n)
        self._Pc_i = np.ones(n)
        self._Vc_i = np.ones(n)
        self._w_i = np.zeros(n)
        self._dipole_i = np.zeros(n)
        self._kappa_i = np.zeros(n)
        self._isGasTransport = np.zeros(n, dtype=bool)
        self._isLiqTransport = np.zeros(n, dtype=bool)

        unknown = []
        for i, sp in enumerate(self.species):
            if sp in FLUID_DATA:
                d = FLUID_DATA[sp]
                self._Tc_i[i] = d['Tc']
                self._Pc_i[i] = d['Pc']
                self._Vc_i[i] = d['Vc']
                self._w_i[i] = d['w']
                self._dipole_i[i] = d['mu']
                self._kappa_i[i] = d['kappa']
                self._isGasTransport[i] = True
            elif sp in _LIQUID_TRANSPORT_SPECIES:
                self._isLiqTransport[i] = True
                base = FLUID_DATA[sp[:-3]]
                self._Tc_i[i], self._Pc_i[i] = base['Tc'], base['Pc']
            else:
                unknown.append(sp)

        self._hasTransportData = self._isGasTransport | self._isLiqTransport

        if unknown:
            print(f"Warning: no transport (viscosity/thermal conductivity) data for "
                  f"species {unknown}; mu and k will exclude {'them' if len(unknown)>1 else 'it'} "
                  f"from the mixing rule (as if not present).")

    def calcTransport(self):
        '''Pure-species dynamic viscosity (mu_i, Pa.s) and thermal conductivity
        (k_i, W/(m.K)) at the current temperature. Gas species use the Chung
        low-pressure-gas viscosity correlation and the Eucken/modified-Eucken thermal
        conductivity correlation; the liquid pseudo-species use dedicated correlations.'''

        self.mu_i = np.zeros(self.Ns)
        self.k_i = np.zeros(self.Ns)

        if np.any(self._isGasTransport):
            g = self._isGasTransport
            self.mu_i[g] = _chungViscosityArr(self._T, self._MWi[g], self._Tc_i[g],
                                               self._Vc_i[g], self._w_i[g],
                                               self._dipole_i[g], self._kappa_i[g])
            cp_mass_g = self.cp_i[g]*1e6/self._MWi[g]   # J/(kg.K)
            R_mass_g = self.R*1e6/self._MWi[g]          # J/(kg.K)
            self.k_i[g] = _euckenConductivityArr(self.mu_i[g], cp_mass_g, R_mass_g, self.condMethod)

        for i in np.nonzero(self._isLiqTransport)[0]:
            sp = self.species[i]
            if sp == 'H2O(L)':
                self.mu_i[i] = _waterLiquidViscosity(self._T)
                self.k_i[i] = _waterLiquidConductivity(self._T)
            elif sp == 'CH3OH(L)':
                self.mu_i[i] = _methanolLiquidViscosity(self._T)
                self.k_i[i] = _methanolLiquidConductivity(self._T)

        # species with no transport data get a tiny, non-zero placeholder viscosity so the
        # Wilke/Mason-Saxena interaction parameters stay finite; they are still excluded
        # from the actual mixture average via the Xi mask applied in mu/k below.
        noData = ~self._hasTransportData
        if np.any(noData):
            self.mu_i[noData] = 1e-12
            self.k_i[noData] = 1e-12
            
    @property
    def mu(self) -> float:
        '''
        Dynamic viscosity of the mixture / Pa.s
        (low-pressure-gas assumption; Chung pure-species values combined with Wilke's mixing rule). Species without
        transport data (see setupTransport) are excluded, as if not present.
        '''
        self.calcTransport()
        Xi_eff = np.where(self._hasTransportData, self._Xi, 0.)
        return _wilkeMixViscosity(Xi_eff, self.mu_i, self._MWi)
    
    @property
    def k(self) -> float:
        '''
        Thermal conductivity of the mixture / W/(m.K)
        (low-pressure-gas assumption; Eucken/modified-Eucken pure-species values combined with the Mason-Saxena
        mixing rule). Species without transport data (see setupTransport) are excluded,
        as if not present.
        '''
        self.calcTransport()
        Xi_eff = np.where(self._hasTransportData, self._Xi, 0.)
        return _masonSaxenaMixConductivity(Xi_eff, self.k_i, self._Tc_i, self._Pc_i, self._MWi, self._T)
    
    @property
    def cp_mass(self) -> float:
        '''Mixture specific heat capacity at constant pressure, mass basis / J/(kg.K)'''
        return self.cp*1e6/self.getMW()
    
    @property
    def Pr(self) -> float:
        '''Mixture Prandtl number / -'''
        return self.mu*self.cp_mass/self.k
    
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
    def mu(self) -> float:
        '''Dynamic viscosity of the mixture / Pa.s'''
        try:
            return self.__AS.viscosity()
        except Exception as e:
            print(f"Warning: CoolProp could not compute viscosity for this state/EOS "
                  f"({self.EOS}): {e}. Returning NaN.")
            return np.nan
        
    @property
    def k(self) -> float:
        '''Thermal conductivity of the mixture / W/(m.K)'''
        try:
            return self.__AS.conductivity()
        except Exception as e:
            print(f"Warning: CoolProp could not compute thermal conductivity for this "
                  f"state/EOS ({self.EOS}): {e}. Returning NaN.")
            return np.nan

    @property
    def Pr(self) -> float:
        '''Mixture Prandtl number / -'''
        return self.mu*self.cp/self.getMW()*1e3/self.k

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
