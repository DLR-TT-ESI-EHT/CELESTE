'''
Flow related objects i.e. sources, sinks and pipes
'''
#from CELESTE2 import 
from . import interfaces, ModelBase
import numpy as np
import scipy.optimize as spo

import numpy.typing as npt
import typing


		
class SourceM(ModelBase):
    '''
    1+nSpecies unknowns
    '''
    def __init__(self, name, Medium, F1_n:float=1., F1_Xi:npt.NDArray[float] | None =None, F1_T:float=298.15, F1_P:float=1.):
        super().__init__(locals())
        self.name = f'{name}(({name}))'
            
        self.F1 = interfaces.Flange(self.name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P)        
        
class Sink(ModelBase):
    '''
    2*(1+nSpecies) unknowns
    '''
    def __init__(self, name, Medium, F0_n:float=1., F0_Xi:npt.NDArray[float] | None=None, F0_T:float=298.15, F0_P:float=1.):
        super().__init__(locals())

        self.name = f'{name}[({name})]'
        self.F0 = interfaces.Flange(self.name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P)
    
class Splitter(ModelBase):
    
    def __init__(self, name:str, Medium, Split:float=0.5, T:float=298.15, P:float=1.,
                 F0_n=1, F0_Xi=None, 
                 F1_n=1, F1_Xi=None, 
                 F2_n=1, F2_Xi=None,               
                ):
        super().__init__(locals())
        
        self.F0: interfaces.Flange = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P)
        self.F1: interfaces.Flange = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P)
        self.F2: interfaces.Flange = interfaces.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=T, P=P)
        self._split: float = Split

    @property
    def split(self) -> float:
        return self._split

    @split.setter
    def split(self, val):
        self._split=val
        
    def Eqs(self):
        self.F1.Medium.PTX = self.F0.Medium.PTX  
        self.F2.Medium.PTX = self.F0.Medium.PTX 
        self.F1.n = self.split*self.F0.n
        self.F2.n = (1-self.split)*self.F0.n
    
    def MBcheck(self):
        # Mass balance
        MB = self.F0.m - self.F1.m - self.F2.m
        print(f"{abs((self.F0.m/(+ self.F1.m+self.F2.m)) - 1)*100}% difference ")    
        return MB
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F2.n*self.F2.Ex - self.F1.n*self.F1.Ex 
    
    @property
    def Sgen(self):
        return self.F2.s*self.F2.n + self.F1.s*self.F1.n - self.F0.s*self.F0.n
    
    def run_sol(self): 
        return self.Eqs()

    
class Mixer(ModelBase):
    
    def __init__(self, name, Medium, T=298.15, P=1.,
                 F0_n:float=1, F0_Xi=None, 
                 F1_n:float=1, F1_Xi=None,            
                ):
        super().__init__(locals())
        
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P)
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium, T=T, P=P)
        
    def Eqs(self):
        self.F2.n = (self.F0.n + self.F1.n)
        self.F2.Xi = (self.F1.Xi*self.F1.n + self.F0.Xi*self.F0.n)/self.F2.n
        self.F2.P = self.F0.P 
        self.Eq = np.array([self.Q/self.F2.n])
    
        return self.Eq

    @property
    def Q(self):
        return self.F2.h*self.F2.n - (self.F1.h*self.F1.n + self.F0.h*self.F0.n)

    def MBcheck(self):
        # Mass balance
        MB = self.F0.m + self.F1.m - self.F2.m
        print(f"{abs(((self.F0.m + self.F1.m)/self.F2.m) - 1)*100}% difference ")    
        return MB

    @property  
    def Exloss(self):
        return (self.F0.n*self.F0.Ex + self.F1.n*self.F1.Ex) - self.F2.n*self.F2.Ex 
    
    @property
    def Sgen(self):
        return self.F2.s*self.F2.n - self.F1.s*self.F1.n - self.F0.s*self.F0.n 
    
    def f_sol(self, x):
        self.F2.T = x[0]
        return self.Eqs()

    def run_sol(self): 
        # Bounds: outlet T must be between inlet temperatures
        T_min = min(self.F0.T, self.F1.T) * 0.95  # Allow 5% margin
        T_max = max(self.F0.T, self.F1.T) * 1.05
        
        # Initial guess: enthalpy-weighted average
        if not np.isclose(self.F0.T, self.F1.T):
            x0 = np.array([(self.F0.T * self.F0.n + self.F1.T * self.F1.n) / (self.F0.n + self.F1.n)])
            x0_clipped = np.clip(x0, T_min, T_max)
            
            sol = spo.least_squares(self.f_sol, x0_clipped, bounds=([T_min], [T_max]))
            
            res = self.f_sol(sol.x)
            if not sol.success or np.linalg.norm(res) > 1e-5:
                print(f'{self.name} Failed: residual={np.linalg.norm(res):.2e}, '
                    f'T=[{self.F0.T:.1f}, {self.F1.T:.1f}] → {self.F2.T:.1f}')
    
            return sol
        else:
            self.f_sol([self.F0.T])
            return 1


