'''
Medium objects to store species info
'''
import numpy as np
from . import thermo

from numpy.typing import NDArray as npArray

# 1. Medium_Base - Base object -> hold the structure
# 2. Fuel_CH4 - fuel side species
# 3. Steam - steam
# 4. Air - normal air species



class Medium_Base:
    def __init__(self, name, species:list[str]= [], reference_X=None):
        self.name = name
        self.species = species #empty list
        self.nX = len(self.species) 
        self._reference_X = reference_X
    
    def __repr__(self):
        return f'{self.name}(species={self.species})' 
    
    def get_Idx(self, species:str):
        return np.nonzero(self.species==species)[0][0]
    

#%% NASA Media
class Medium_NASAbase(Medium_Base, thermo.Thermo):
    """NASA polynomial-based medium."""
    
    _default_species = None
    _default_Xi = None
    
    def __init__(self, name:str=' ', species=None, T=298.15, Xi:npArray[np.float64]|None=None, P=1.):
        species = species if species is not None else self._default_species
        Xi = Xi if Xi is not None else self._default_Xi
        Medium_Base.__init__(self, name, species, reference_X=Xi) 
        thermo.Thermo.__init__(self, species=species, T=T, Xi=Xi, P=P)
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        if cls._default_species is not None:
            species_str = ', '.join(cls._default_species)
            Xi_str = np.array2string(cls._default_Xi, precision=2, separator=', ') if cls._default_Xi is not None else 'None'
            
            base_doc = cls.__doc__ or ''
            cls.__doc__ = f"""{base_doc}
    
            Default Species: {species_str}
            Default Composition: {Xi_str}
            """

    def __copy__(self):
        return type(self)()
    
    @property
    def rho(self):
        """Density in kg/m³"""
        raise NotImplementedError("Subclasses must implement rho property")

class Fuel_CH4(Medium_NASAbase): #extending medium
    _default_species = ['H2','CH4','CO2','CO','H2O','N2']
    _default_Xi = np.array([0.1, 1e-8, 1e-8, 1e-8, 0.9, 1e-8])
    def __init__(self, name = 'Fuel_CH4'):
        super().__init__(name) 

    @property
    def rho(self):
        return self.P*1e5*self.getMW()*1e-3/(self.R*1e3)/self.T
        
class Fuel_CH4L(Medium_NASAbase): #extending medium
    _default_species = ['H2','CH4','CO2','CO','H2O(L)','N2']
    _default_Xi = np.array([0.1, 1e-8, 1e-8, 1e-8, 0.9, 1e-8])
    def __init__(self, name = 'Fuel_CH4L'):
        super().__init__(name) 

    @property
    def rho(self):
        #rho = 0.325*0.27**(-(1 - self.F0.T/647.13)**0.23)
        rho = 0.3471*0.274**-(1 - (self.T/647.13))**0.28571 
        return rho*1e3  # Kg/m3

class Steam(Medium_NASAbase): #extending medium
    _default_species = ['H2O']
    _default_Xi = np.array([1.0])
    def __init__(self, name = 'Steam'):
        super().__init__(name) 
    
    @property
    def rho(self):
        return self.P*1e5*self.getMW()*1e-3/(self.R*1e3)/self.T
        
class Air(Medium_NASAbase): 
    _default_species = ['O2', 'N2']
    _default_Xi = np.array([0.21, 0.79])
    def __init__(self, name = 'Air'):
        super().__init__(name)

    @property
    def rho(self):
        return self.P*1e5*self.getMW()*1e-3/(self.R*1e3)/self.T
        
class Water(Medium_NASAbase):
    _default_species = ['H2O(L)']
    _default_Xi = np.array([1.0])
    def __init__(self, name = 'Water'):
        super().__init__(name)
    
    @property
    def rho(self):
        #rho = 0.325*0.27**(-(1 - self.F0.T/647.13)**0.23)
        rho = 0.3471*0.274**-(1 - (self.T/647.13))**0.28571 
        return rho*1e3  # Kg/m3
        
class HB(Medium_NASAbase): #extending medium
    _default_species = ['H2','N2','NH3']
    _default_Xi = np.array([0.1, 0.9, 1e-8])
    def __init__(self, name = 'H_B'):
        super().__init__(name)

    @property
    def rho(self):
        return self.P*1e5*self.getMW()*1e-3/(self.R*1e3)/self.T
             
class Steam_Water(Medium_NASAbase): #extending medium
    _default_species = ['H2O', 'H2O(L)']
    _default_Xi = np.array([0.97, 0.03])
    def __init__(self, name = 'Steam'):
        super().__init__(name)
    
    @property
    def rho(self):
        #rho = 0.325*0.27**(-(1 - self.F0.T/647.13)**0.23)
        rho = 0.3471*0.274**-(1 - (self.T/647.13))**0.28571 
        return rho*1e3  # Kg/m3
        
class Fuel_MeOH(Medium_NASAbase): #extending medium
    _default_species = ['H2','CH4','CO2','CO','H2O','N2', 'CH3OH']
    _default_Xi = np.array([0.2, 1e-8, 0.05, 0.05, 0.3, 1e-8, 0.4])
    def __init__(self, name = 'Fuel_MeOH'):
        super().__init__(name) 

    @property
    def rho(self):
        return self.P*1e5*self.getMW()*1e-3/(self.R*1e3)/self.T

class Fuel_MeOHL(Medium_NASAbase): #extending medium 
    _default_species = ['H2','CH4','CO2','CO','H2O(L)','N2', 'CH3OH(L)']
    _default_Xi = np.array([0.2, 1e-8, 0.05, 0.05, 0.3, 1e-8, 0.4])
    def __init__(self, name = 'Fuel_MeOHL'):
        super().__init__(name)

    @property
    def rho(self):
        return 1/(self.Yi[4]/self.rho_water() + self.Yi[6]/self.rho_MeOH())
    
    def rho_water(self):
        # the one working https://booksite.elsevier.com/9780750683661/Appendix_C.pdf
        ## the uncommented are from The Yaws handbook of physical properties
        ## T in K
        #rho = 0.325*0.27**(-(1 - self.F0.T/647.13)**0.23)
        rho = 0.3471*0.274**-(1 - (self.T/647.13))**0.28571 #self.Tav
        return rho*1e3  # Kg/m3
    
    def rho_MeOH(self):
        rho = 0.27197*0.27192**-(1 - (self.T/512.58))**0.23310 #self.Tav
        return rho*1e3  # Kg/m3

class Media_AEL(Medium_NASAbase):
    _default_species = ['H2O(L)', 'H2', 'O2', 'H2O']
    _default_Xi = np.array([0.997, 1e-8, 1e-8, 0.003])
    def __init__(self, name='Media_AEL'):
        super().__init__(name)

    @property
    def rho(self):
        #rho = 0.325*0.27**(-(1 - self.F0.T/647.13)**0.23)
        rho = 0.3471*0.274**-(1 - (self.T/647.13))**0.28571 
        return rho*1e3  # Kg/m3

class media_C3(Medium_NASAbase):
    _default_species = ['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2', 'C3H8']
    _default_Xi = np.array([0.1, 1e-5, 1e-5, 1e-5, 0.9, 1e-5, 1e-5])
    def __init__(self, name = 'media_C3'):
        super().__init__(name) 

    @property
    def rho(self):
        return self.P*1e5*self.getMW()*1e-3/(self.R*1e3)/self.T

class Atm_Air(Medium_NASAbase): 
    _default_species = ['O2', 'N2', 'CO2']
    _default_Xi = np.array([0.2114, 0.7883, 0.0003])
    def __init__(self, name = 'Atm_Air'):
        super().__init__(name)

    @property
    def rho(self):
        return self.P*1e5*self.getMW()*1e-3/(self.R*1e3)/self.T

#%% CP Media
class Medium_CPbase(Medium_Base, thermo.Thermo_CP):
    def __init__(self, name, species=[], T=298.15, Xi=None, P=1., phase='gas', EOS='HEOS'):
        Medium_Base.__init__(self, name, species, Xi) 
        thermo.Thermo_CP.__init__(self, species=species, T=T, Xi=Xi, P=P, EOS=EOS, phase=phase)
        self.phase = phase
        self.EOS = EOS

    def __copy__(self):
        return type(self)(self.name, self.phase, self.EOS)

    @property
    def rho(self):
        return self.P*1e5*self.getMW()*1e-3/(self.R*1e3)/self.T # needs to be implemented from coolprop

class Fuel_CH4_CP(Medium_CPbase): #extending medium
    def __init__(self, name = 'Fuel_CH4_CP', phase='gas', EOS='HEOS'):
        super().__init__(name, species=['H2','CH4','CO2','CO','H2O','N2'], Xi=np.array([0.1, 1e-8, 1e-8, 1e-8, 0.9, 1e-8]), phase=phase, EOS=EOS) 
