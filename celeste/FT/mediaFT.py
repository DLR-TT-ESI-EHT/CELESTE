import numpy as np

from ..CSL import media
from . import thermoFT as thFT

from typing import Literal
from numpy.typing import NDArray as npArray


class Medium_Syncrude(media.Medium_Base, thFT.Thermo):
    def __init__(self, name:str, species:dict={}, T:float=298.15, Xi:npArray[np.float64]|None=None, P:float=1., phase:Literal['Gas', 'Liquid']='Gas'):
        tot_species = [
            'H2', 'CO2', 'CO', 'H2O', 'N2', 'CH4', 'C2H6', 'C3H8', 'C4H10',
           'C5H12', 'C6H14', 'C7H16', 'C8H18', 'C9H20', 'C10H22', 'C11H24',
           'C12H26', 'C13H28', 'C14H30', 'C15H32', 'C16H34', 'C17H36',
           'C18H38', 'C19H40', 'C20H42', 'C21H44', 'C22H46', 'C23H48',
           'C24H50', 'C25H52', 'C26H54', 'C27H56', 'C28H58', 'C29H60',
           'C30H62', 'C31H64', 'C32H66', 'C33H68', 'C34H70', 'C35H72',
           'C36H74', 'C37H76', 'C38H78', 'C39H80', 'C40H82', 'C41H84',
           'C42H86', 'C43H88', 'C44H90', 'C45H92', 'C46H94', 'C47H96',
           'C48H98', 'C49H100', 'C50H102']

        self.I1 = species['1']
        self.I2 = species['2']
        sp = tot_species[self.I1:self.I2]
        media.Medium_Base.__init__(self, name, sp)
        thFT.Thermo.__init__(self, species=sp, T=T, Xi=Xi, phase=phase, P=P)

class Water(Medium_Syncrude): #extending medium
    def __init__(self, name = 'Water'):
        self._reference_X = np.array([1.]) # try with 1.0
        super().__init__(name, species={'1':3, '2':4}, Xi=self._reference_X) 
        
class Hydrogen(Medium_Syncrude):
    def __init__(self, name = 'Hydrogen'):
        self._reference_X = np.array([1])
        super().__init__(name, species={'1':0, '2':1}, Xi=self._reference_X) 
        
class Alkanes(Medium_Syncrude):
    def __init__(self, name = 'Alkanes'):
        self._reference_X = np.concatenate((
            [0,                      0,  5.00591288e-01,  3.68163327e-01,
             0,            9.41777835e-02,  1.01813820e-02,  2.54534550e-03,
             1.27267275e-03,  4.85606625e-03,  3.64204969e-03,  2.80958119e-03,
             2.21254518e-03,  1.77003615e-03,  1.43372928e-03,  1.17305123e-03,
             9.67767264e-04,  8.03991265e-04,  6.71906986e-04,  5.64401868e-04,
             4.76214076e-04,  4.03381335e-04,  3.42874135e-04,  2.92345315e-04,
             2.49955244e-04,  2.14247352e-04,  1.84057953e-04], np.zeros(23)))
        super().__init__(name, species={'1':5, '2':55}, Xi=self._reference_X)
        
class LightGas(Medium_Syncrude):
    def __init__(self, name = 'Light'):
        self._reference_X = np.array([0.25, 0.25, 0.25, 0.25])
        super().__init__(name, species={'1':5, '2':9}, Xi=self._reference_X) # C1-4 
        
class Naphtha(Medium_Syncrude):
    def __init__(self, name = 'Naphtha'):
        self._reference_X = np.array([0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667]) # change this
        super().__init__(name, species={'1':9, '2':15}, Xi=self._reference_X) # C5-10
        
class Kerosene(Medium_Syncrude):
    def __init__(self, name = 'Kerosene'):
        self._reference_X = np.array([0.25, 0.25, 0.25, 0.25])
        super().__init__(name, species={'1':15, '2':19}, Xi=self._reference_X) # C11-C14
        
class Distillate(Medium_Syncrude):
    def __init__(self, name = 'Distillate'):
        self._reference_X = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
        super().__init__(name, species={'1':19, '2':27}, Xi=self._reference_X) # C15-C22
        
class Wax(Medium_Syncrude):
    def __init__(self, name = 'Wax'):
        self._reference_X = np.array((
            [0.13398714, 0.11556391, 0.09984722, 0.08640625, 0.07488541,
             0.06498984, 0.05647393, 0.04913232, 0.04279267, 0.03730986,
             0.03256133, 0.02844328, 0.02486755, 0.02175911, 0.01905392,
             0.01669725, 0.01464221, 0.01284854, 0.01128164, 0.00991173,
             0.0087131 , 0.00766357, 0.00674394, 0.0059376 , 0.00523014,
             0.00460906, 0.0040635 , 0.00358401]))
        super().__init__(name, species={'1':27, '2':55}, Xi=self._reference_X) # C23-C50
        
class Syncrude(Medium_Syncrude):
    def __init__(self, name = 'Syncrude'):
        self._reference_X = np.concatenate((
            [0,                      0,  5.00591288e-01,  3.68163327e-01,
             0,            9.41777835e-02,  1.01813820e-02,  2.54534550e-03,
             1.27267275e-03,  4.85606625e-03,  3.64204969e-03,  2.80958119e-03,
             2.21254518e-03,  1.77003615e-03,  1.43372928e-03,  1.17305123e-03,
             9.67767264e-04,  8.03991265e-04,  6.71906986e-04,  5.64401868e-04,
             4.76214076e-04,  4.03381335e-04,  3.42874135e-04,  2.92345315e-04,
             2.49955244e-04,  2.14247352e-04,  1.84057953e-04], np.zeros(28)))
        super().__init__(name, species={'1':0, '2':55}, Xi=self._reference_X)
        
        self._reference_X2 = np.concatenate(( np.zeros(27), 
            [0.13398714, 0.11556391, 0.09984722, 0.08640625, 0.07488541,
             0.06498984, 0.05647393, 0.04913232, 0.04279267, 0.03730986,
             0.03256133, 0.02844328, 0.02486755, 0.02175911, 0.01905392,
             0.01669725, 0.01464221, 0.01284854, 0.01128164, 0.00991173,
             0.0087131 , 0.00766357, 0.00674394, 0.0059376 , 0.00523014,
             0.00460906, 0.0040635 , 0.00358401]))
        
        
     
