import copy

import numpy as np
from numpy.typing import NDArray as npArray

from ..CSL import interfaces
from . import mediaFT

class Flange(interfaces.Flange): 
    def __init__(self, name:str, Medium:mediaFT.Medium_Syncrude, n:float=1, Xi:npArray[np.float64]|None=None, T:float=298.15, P:float=1, phase:str='Gas'):
        Medium=copy.deepcopy(Medium)
        #self.Medium.phase = phase
        super().__init__(name, Medium, n, Xi, T, P)
        if phase in ['gas', 'Gas']:
            self.phase = 'Gas'
        elif phase in ['liquid', 'Liquid']:
            self.phase = 'Liquid'
        else:
            print(f'Warning: Wrong phase {phase}')
    
        if Xi is None:
            Xi = self.Medium._reference_X

    @property
    def phase(self):
        return self.Medium.phase

    @phase.setter
    def phase(self, pha):
        self.__dict__['phase'] = pha
        self.Medium.phase = pha
