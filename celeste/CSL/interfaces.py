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

    @property
    def mw(self):
        '''
        Molar Mass /g/mol
        '''
        return self.Medium.getMW()

    @property
    def cp(self):
        '''
        Specific Heat Capacity /kJ/mol/K
        '''
        return self.Medium.cp
    
    @property
    def h(self):
        '''
        Specific Enthalpy /kJ/mol
        '''
        return self.Medium.h

    @property
    def s(self):
        '''
        Specific Entropy /kJ/mol/K
        '''
        return self.Medium.s

    @property
    def g(self):
        '''
        Specific Gibbs Energy /kJ/mol
        '''
        return self.Medium.g

    @property
    def T(self):
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
    def P(self):
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
    def m(self):
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
    def Ex(self):
        return self.ExergyObj.Ex
    
    @property
    def LHV(self):
        return self.Medium.LHVmix

    def T_hx(self, h, X):
     
        self.Xi = X
        
        def loss(t, h_args=h):
            self.T = t[0]
            return np.abs(self.h - h_args)
        
        sol = spo.least_squares(loss, self.T, bounds=(273.15, 2000.), args=(h,))
        self.T = sol.x[0]

    def T_sPX(self, s, P, X):
        self.P = P
        self.Xi = X

        def loss(T, s):
            self.T = T[0]
            res = np.abs(self.s*1e3 - s*1e3)
            return res
        
        sol = spo.least_squares(loss, self.T, bounds=(273.15, 2000.), args=(s,))
        self.T = sol.x[0] 

    def getCarbon(self, T_drop=0):
        return carbonFraction(self.T-T_drop, self.P, dict(zip(self.Medium.species, self.Xi)))
    
