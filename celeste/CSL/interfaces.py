'''
Interfaces
'''
import copy

import numpy as np
from numpy.typing import NDArray as npArray
import pandas as pd
import scipy.optimize as spo

from . import media
from .. import Base
from ..FT import thermoFT as thFT
from ..functions import carbonFraction

class Flange(Base):

    _pin_vars = ['n', 'Xi', 'T', 'P']
    _flow_vars = []
    
    def __init__(self, name:str, Medium:media.Medium_Base, n:float=1, Xi:npArray[np.float64]|None=None, T:float=298.15, P:float=1.):
        Medium=copy.copy(Medium) # to avoid weird behaviour when using the same medium variable for more than one flange
        super().__init__(locals())
        self.Medium: media.Medium_Base = Medium
        #self.name = f"{name}"
        
        if Xi is None:
            Xi  = self.Medium._reference_X

        self.Medium.PTX = P, T, np.array(Xi)
        
        #if np.sum(self.Xi)>1.01:
        #    print(f'Warning: {self.name}.Xi has sum greater than 1:  sum({self.name}.Xi) = {np.sum(self.Xi)}')

        # The superclass Base() already makes the following assignments. But it's repeated here just for the sake of avoiding Pyright warnings/errors.
        self.n: float = n

    @property
    def mw(self) -> float:
        '''
        Molar Mass /g/mol
        '''
        return self.Medium.getMW()

    @property
    def cp(self) -> float:
        '''
        Specific Heat Capacity /kJ/mol/K
        '''
        return self.Medium.cp
    
    @property
    def h(self) -> float:
        '''
        Specific Enthalpy /kJ/mol
        '''
        return self.Medium.h

    @property
    def s(self) -> float:
        '''
        Specific Entropy /kJ/mol/K
        '''
        return self.Medium.s

    @property
    def g(self) -> float:
        '''
        Specific Gibbs Energy /kJ/mol
        '''
        return self.Medium.g
    
    @property
    def rho(self) -> float:
        '''
        Density /kg/m3
        '''
        return self.Medium.rho
    
    @property
    def mu(self) -> float:
        '''
        Dynamic Viscosity /Pa.s
        '''
        return self.Medium.mu
    
    @property
    def k(self) -> float:
        '''
        Thermal Conductivity /W/m/K
        '''
        return self.Medium.k
    
    @property
    def Pr(self) -> float:
        '''
        Prandtl Number /-
        '''
        return self.Medium.Pr

    @property
    def T(self) -> float:
        '''
        Absolute Temperature /K
        '''
        return self.Medium.T
    
    @T.setter
    def T(self, T):
        self.__dict__['T'] = T
        self.Medium.T = T
        
    @property
    def Xi(self) -> npArray[np.float64]:
        '''
        Mole Fractions /-
        '''
        return self.Medium.Xi
    
    @Xi.setter
    def Xi(self, xi):
        if xi is None:
            xi = self.Medium._reference_X
        self.__dict__['Xi'] = xi
        self.Medium.Xi = np.array(xi)
        #if np.sum(self.Xi)>1.01:
        #    print(f'Warning: {self.name}.Xi has sum greater than 1:  sum({self.name}.Xi) = {np.sum(self.Xi)}')
        
    @property
    def P(self) -> float:
        '''
        Pressure /bar
        '''
        return self.Medium.P
    
    @P.setter
    def P(self, Ps):
        self.__dict__['P'] = Ps
        self.Medium.P = Ps
    
    @property
    def Yi(self):
        '''
        Mass Fraction /-
        '''
        return self.Medium.Yi

    @Yi.setter
    def Yi(self, y):
        self.Medium.Yi = y
        self.Xi = self.Medium.Xi

    @property
    def m(self) -> float:
        '''
        Mass Flow /g/s
        '''
        return self.n*self.mw

    @m.setter
    def m(self, massFlow):
        self.n = massFlow/self.mw

    def get_nXTP(self):
        ''' 
        Returns array of [n, Xi, T, P] as a 1 dimensional vector
        '''
        val_arr = np.concatenate(([self.n, self.m, self.T, self.P, self.h, self.cp], self.Xi))
        head_arr = ['n /mol/s', 'm /g/s', 'T /K', 'P /bar', 'h /kJ/mol', 'cp /kJ/mol/K'] + [f'Xi_{species}' for species in self.Medium.species]

        return pd.Series(dict(zip(head_arr, val_arr)), name = self.name )

    def __repr__(self):
        return self.get_nXTP().to_string()
    
    def __eq__(self, other):
        if type(self)==type(other) and type(self.Medium)==type(other.Medium):
            return True
        else:
            return False

    @property
    def ExergyObj(self):
        return thFT.ThermoExergy(self.Medium, self.Xi, self.T, self.P)
        
    @property
    def Ex(self) -> float:
        return self.ExergyObj.Ex
    
    @property
    def LHV(self) -> float:
        return self.Medium.LHVmix
        
    def T_hx(self, h, X, tol=1e-12, maxit=20, Tmin=273.15, Tmax=3000.):
        self.Xi = X
        T = self.T
        for _ in range(maxit):
            self.T = T
            dh = h - self.h
            if abs(dh) < tol:
                return
            T = min(max(T + dh/self.cp, Tmin), Tmax)
        self.T = T 
        
    def T_sPX(self, s, P, X, tol=1e-12, maxit=20, Tmin=273.15, Tmax=3000.):
        self.P = P
        self.Xi = X
        T = self.T
        for _ in range(maxit):
            self.T = T
            ds = s - self.s
            if abs(ds) < tol:
                return
            T = min(max(T*np.exp(ds/self.cp), Tmin), Tmax)
        self.T = T

    def getCarbon(self, T_drop=0):
        return carbonFraction(self.T-T_drop, self.P, dict(zip(self.Medium.species, self.Xi)))
    
