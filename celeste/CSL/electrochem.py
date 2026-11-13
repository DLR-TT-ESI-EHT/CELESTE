'''
Electrochemical reactors
'''

from . import interfaces, ModelBase
import numpy as np
import scipy.optimize as spo
from celeste import functions as funs
from celeste.CSL.media import Medium_NASAbase, AirHumid, Fuel_CH4

# For Pyright type hints
from numpy._typing._array_like import NDArray
from typing import override, Literal

class SOC(ModelBase):
    '''
    
    '''
    def __init__(self, name:str, MediumF:Medium_NASAbase, MediumA:Medium_NASAbase, Tsoc0:float=1073.15, Tsoc1:float=1073.15, I:float=0.0, cellArea:float=100., Ncells:int|float=1,
                 f_ASR:float=1., extent:bool=False, Qloss:float=0.0 ): 
        super().__init__(locals())
        self.F0: interfaces.Flange = interfaces.Flange(name + '.F0', Medium=MediumF, T=Tsoc0)
        self.F1: interfaces.Flange = interfaces.Flange(name + '.F1', Medium=MediumF, T=Tsoc1)
        self.A0: interfaces.Flange = interfaces.Flange(name + '.A0', Medium=MediumA, T=Tsoc0)
        self.A1: interfaces.Flange = interfaces.Flange(name + '.A1', Medium=MediumA, T=Tsoc1)

        self._N_H: NDArray[np.int64] = np.array([2,4,0,0,2,0])
        self._N_C: NDArray[np.int64] = np.array([0,1,1,1,0,0])
        self._N_O: NDArray[np.int64] = np.array([0,0,2,1,1,0])
        self._N_N: NDArray[np.int64] = np.array([0,0,0,0,0,2])

        # The superclass Base() already makes these assignments via super().__init__(locals()). But repeated here just for the sake of pyright.
        self.name: str = name
        self.Tsoc0: float = Tsoc0
        self.Tsoc1: float = Tsoc1
        self.I: float = I
        self.cellArea: float = cellArea
        self.Ncells: float|int = Ncells
        self.f_ASR: float = f_ASR
        self.extent: bool = extent
        self.Qloss: float = Qloss

        # Additional declarations for Pyright
        self._sol: spo.OptimizeResult

    @property
    def P_soc(self) -> float:
        return self.F1.P
    
    @P_soc.setter
    def P_soc(self, P:float):
        self.F1.P = P
        self.A0.P = P
        self.A1.P = P

    @override
    def Eqs(self):
        self.P_soc=self.F0.P
        n_O = self.n_O

        self.Eq: NDArray[np.float64] = np.array([
            # Fuel side atom balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)), #
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)), #
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)) + n_O, #
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)), #

            # Fuel-side equilibria
            self.F1.Xi[3]*self.F1.Xi[4]*self.K_WGS() - self.F1.Xi[2]*self.F1.Xi[0], #
            self.F1.Xi[1]*self.F1.Xi[4]*self.K_MSR() - self.F1.Xi[3]*(self.F1.Xi[0]**3)*(self.F1.P**2)
        ]) 

        # Air side reactions forward-pass, no need for equilibrium
        O2_out: float = self.A0.n*self.A0.Xi[0] - 0.5*n_O
        self.A1.n = self.A0.n - 0.5*n_O
        Xi_O2_out = O2_out/self.A1.n
        self.A1.Xi = np.array([Xi_O2_out, 1-Xi_O2_out], dtype=float)

        return self.Eq
    
    def MBcheck(self) -> float:
        MB = self.F0.m + self.A0.m - self.F1.m - self.A1.m
        print(f'{abs(((self.F0.m + self.A0.m)/(self.F1.m + self.A1.m) -1))*100 }% difference')
        return MB
    
    @property
    def n_O(self) -> float:
        return (self.Ncells*self.I/2/96485)

    @property
    def Wchem(self) -> float:
        if self.extent:
            R = 8.314462618e-3
            gi_p = self.F1.T*R*np.log(self.F1.Xi*self.F1.P, where=(self.F1.Xi>=1e-6))
            return self.MSR_extent*(
                        4*(self.F1.Medium.g_i[0] + gi_p[0]) + (self.F1.Medium.g_i[2] + gi_p[2]) 
                        - 2*(self.F1.Medium.g_i[4] + gi_p[4]) - (self.F1.Medium.g_i[1] + gi_p[1]))
        else:
             return 0.
    
    @property
    def MSR_extent(self) -> float:
        return (self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1])

    @property
    def Wid(self) -> float:
        return (self.F1.n*self.F1.g + self.A1.n*self.A1.g - self.F0.n*self.F0.g - self.A0.n*self.A0.g) - self.Wchem

    @property
    def Vid(self) -> float:
        return self.Wid*1e3/(-self.I*self.Ncells)

    @property
    def j(self) -> float:
        return self.I/self.cellArea

    @j.setter
    def j(self, val:float):
        self.I = val*self.cellArea

    @property
    def Vop(self) -> float:
        return self.Vid - self.f_ASR*self.ASR()*self.j

    @property
    def W(self) -> float:
        return self.Vop*(-self.I)*self.Ncells*1e-3
        
    @property
    def Q(self) -> float:
        return self.F1.n*self.F1.h + self.A1.n*self.A1.h - self.F0.n*self.F0.h - self.A0.n*self.A0.h - self.W - self.Qloss 
    
    @property
    def Exth(self) -> float:
        # Qloss = 0, change Exth when Qloss is not zero
        return self.Q*(1 - (298.15/self.F1.T))  
    
    @property
    @override
    def Exloss(self) -> float:
        return self.F0.Ex*self.F0.n + self.A0.Ex*self.A0.n - self.F1.Ex*self.F1.n - self.A1.Ex*self.A1.n + self.W + self.Exth 
        
    @property
    def RC(self) -> float:
        if self.I < 0:
            val = -self.Ncells*self.I/(96485*(self.F0.n*(2*self.F0.Xi[2] + 2*self.F0.Xi[4]) - (2/self.RC_EC*self.MSR_extent if self.extent else 0) ) ) 
        else:
            val = self.Ncells*self.I/(2*96485*self.F0.n*(self.F0.Xi[0] + 4*self.F0.Xi[1] + self.F0.Xi[3]))
        return val
    
    @property
    def RC_EC(self) -> float:
        return (self.F0.n*(self.F0.Xi[2] + self.F0.Xi[4]) - self.F1.n*(self.F1.Xi[2] + self.F1.Xi[4]))/(
                self.F0.n*(self.F0.Xi[2] + self.F0.Xi[4]))

    @property
    def AU(self) -> float:
        return 1 - (self.A1.Xi[0]*self.A1.n / self.A0.Xi[0]/self.A0.n)
    
    def K_WGS(self) -> float:
        return np.exp(-(32.1153*(self.F1.T) - 3.5211e4)/(8.314*self.F1.T)) 
    
    def K_MSR(self) -> float:
        return np.exp(-(225215.699 - 252.643*self.F1.T)/(8.314*self.F1.T))       
    
    def ASR(self) -> float:
        # M. Riedel 2020 DOI 10.1149/1945-7111/ab6820
        return 7.28e2*np.exp(-8.29e-3*((self.F1.T)-273.15))

    def cform(self, side:str='out'):
        if side == 'out':
            f = self.F1
        else:
            f = self.F0
        return funs.carbonFraction(f.T, f.P, dict(zip(f.Medium.species, f.Xi)))

    @property
    def Sgen(self) -> float:
        return self.F1.s*self.F1.n + self.A1.s*self.A1.n - self.F0.s*self.F0.n - self.A0.s*self.A0.n - self.Q/self.F1.T

    @property
    def EnthalpyEfficiency(self) -> float:
        LHV_H2: float = self.F1.Medium.h_i[4] - self.F1.Medium.h_i[0] - 0.5*self.A1.Medium.h_i[0]
        LHV_CO: float = self.F1.Medium.h_i[2] - self.F1.Medium.h_i[3] - 0.5*self.A1.Medium.h_i[0]
        LHV_CH4: float = self.F1.Medium.h_i[2] + 2*self.F1.Medium.h_i[4] - self.F1.Medium.h_i[3] - 2*self.A1.Medium.h_i[0]
        LHV_F0: float = np.dot(-np.array([LHV_H2, LHV_CH4, 0, LHV_CO, 0, 0]), self.F0.Xi)*self.F0.n
        LHV_F1: float = np.dot(-np.array([LHV_H2, LHV_CH4, 0, LHV_CO, 0, 0]), self.F1.Xi)*self.F1.n
        return (LHV_F1 - LHV_F0)/self.W

    @property
    def Lambda(self) -> float:
        if self.I > 0:
            val:float = (self.A0.n*self.A0.Xi[0])/(2*self.n_O)
        else:
            val = np.nan
        return val
        
    def f_sol(self, x:NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.abs(x)
        self.F1.Xi =  np.concatenate((x[:5], [1-np.sum(x[:5])]))
        self.F1.n = x[5]

        return self.Eqs()
        
    def run_sol(self, x0:NDArray[np.float64]|None=None):
        if x0 is None:
            x0 = np.concatenate((self.F0.Xi[:5], [self.F0.n]))

        lower_bounds = [1e-8,1e-8,1e-8,1e-8,1e-8,1e-8*self.Ncells]
        upper_bounds = [0.9999,0.9999,0.9999,0.9999,0.9999,1e1*self.Ncells]
        x0_clipped = np.clip(np.abs(x0), lower_bounds, upper_bounds)

        sol = spo.least_squares(self.f_sol, x0_clipped, xtol=1e-15, bounds = (lower_bounds, upper_bounds)) 
        if np.linalg.norm(sol.fun)<=1e-7:
            sol = spo.root(self.f_sol, sol.x)
        if np.linalg.norm(sol.fun)>1e-7:
            sol = spo.root(self.f_sol, x0)
        _ = self.f_sol(sol.x) # Call the function once to make sure the system is in the optimised state (and not at the last solver iteration)
        if not sol.success:
            print(f'{self.name} Failed')
        self._sol = sol
        return sol  
    


class SOC_Nernst_avg(SOC):
    '''
    SOC model that uses Nernst equation with average flow compositions
    '''

    @property
    def Vnernst(self) -> float:
        R = 8.314462618
        E0: float = ((self.F1.Medium.g_i[4] - self.F1.Medium.g_i[0]- 0.5*self.A1.Medium.g_i[0])*1e3/2/96485) 
        aver: NDArray[np.float64] = np.mean([[self.F0.Xi[0], self.F1.Xi[0]], [self.F0.Xi[4], self.F1.Xi[4]], [self.A0.Xi[0], self.A1.Xi[0]]], axis=1)  

        _x_avg_names = ["H2" ,"H2O", "O2"]
        for index in np.where(aver < 1e-6)[0]:
            print(f"Mole fraction {_x_avg_names[index]} is (almost) 0") 

        Eid = -E0 - R*self.A1.T/2/96485*(np.log(aver[1])-np.log(aver[0])-0.5*np.log(self.P_soc*aver[2]))
        return Eid

    @property
    @override
    def Wid(self) -> float:
        return (-self.Vnernst*self.j*self.cellArea*self.Ncells/1e3) - self.Wchem



class PCC_base(ModelBase):

    def __init__(self, name:str, Tpcc0:float=873.15, Tpcc1:float=873.15, I:float=1., cellArea:float=100000., Ncells:int|float=1,
                 f_ASR:float=1., extent:bool=False, Qloss:float=0., EffFar:float=1.):
        super().__init__(locals())
        
        MediumF = Fuel_CH4()
        MediumA = AirHumid()
        self.F0: interfaces.Flange = interfaces.Flange(name + '.F0', Medium=MediumF, T=Tpcc0)
        self.F1: interfaces.Flange = interfaces.Flange(name + '.F1', Medium=MediumF, T=Tpcc1)
        self.A0: interfaces.Flange = interfaces.Flange(name + '.A0', Medium=MediumA, T=Tpcc0)
        self.A1: interfaces.Flange = interfaces.Flange(name + '.A1', Medium=MediumA, T=Tpcc1)

        self._N_H: NDArray[np.int64] = np.array([2,4,0,0,2,0])
        self._N_C: NDArray[np.int64] = np.array([0,1,1,1,0,0])
        self._N_O: NDArray[np.int64] = np.array([0,0,2,1,1,0])
        self._N_N: NDArray[np.int64] = np.array([0,0,0,0,0,2])

        # The superclass Base() already makes these assignments via super().__init__(locals()). But repeated here just for the sake of pyright.
        self.name: str = name
        self.Tpcc0: float = Tpcc0
        self.Tpcc1: float = Tpcc1
        self.I: float = I
        self.cellArea: float = cellArea
        self.Ncells: float|int = Ncells
        self.f_ASR: float = f_ASR
        self.extent: bool = extent
        self.Qloss: float = Qloss
        self.EffFar: float = EffFar
        
        # Additional declarations for Pyright
        self._sol: spo.OptimizeResult

    @property
    def P_pcc(self) -> float:
        return self.F1.P

    @P_pcc.setter 
    def P_pcc(self, P:float): 
        self.F1.P = P
        self.A0.P = P
        self.A1.P = P

    @override
    def Eqs(self) -> NDArray[np.float64]:
        self.P_pcc = self.F0.P 
        n_H = self.n_H  

        self.Eq: NDArray[np.float64] = np.array([
            # Fuel side atom balances
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)),
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)) - n_H, # Protons move from FE to AE.
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)),
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)),

            # Fuel side equilibria
            self.F1.Xi[3]*self.F1.Xi[4]*self.K_WGS() - self.F1.Xi[2]*self.F1.Xi[0],
            self.F1.Xi[1]*self.F1.Xi[4]*self.K_MSR() - self.F1.Xi[3]*(self.F1.Xi[0]**3)*(self.F1.P**2)          
        ])

        # Air side reactions, no need for equilibrium
        O2_out: float = self.A0.n*self.A0.Xi[0] - 0.25*n_H
        H2O_out: float = self.A0.n*self.A0.Xi[2] + 0.5*n_H 
        self.A1.n = self.A0.n + 0.25*n_H
        Xi_O2_out = O2_out/self.A1.n
        Xi_H2O_out = H2O_out/self.A1.n
        self.A1.Xi = np.array([Xi_O2_out, 1-Xi_O2_out-Xi_H2O_out, Xi_H2O_out], dtype=float)
        
        return self.Eq

    def MBcheck(self) -> float:
        MB = self.F0.m + self.A0.m - self.F1.m - self.A1.m
        print(f'{abs(((self.F0.m + self.A0.m)/(self.F1.m + self.A1.m) -1))*100 }% difference')
        return MB

    @property # mol/s of H+ (not H2) crossing the electrolyte
    def n_H(self) -> float:
        if self.I < 0:
            nH = self.Ncells*self.I*self.EffFar/96485 
        else: 
            nH = self.Ncells*self.I/self.EffFar/96485
        return nH

    @property
    def Wchem(self) -> float:
        if self.extent:
            R = 8.314462618e-3
            gi_p = self.F1.T*R*np.log(self.F1.Xi*self.F1.P, where=(self.F1.Xi>=1e-6))
            return self.MSR_extent*(
                        4*(self.F1.Medium.g_i[0] + gi_p[0]) + (self.F1.Medium.g_i[2] + gi_p[2]) 
                        - 2*(self.F1.Medium.g_i[4] + gi_p[4]) - (self.F1.Medium.g_i[1] + gi_p[1]))
        else:
             return 0.

    @property
    def MSR_extent(self) -> float:
        return (self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1])

    @property
    def Wid(self) -> float:
        return np.nan # This will be redefined in each of the subclasses PCC_Nernst_avg and PCC_delG

    @property
    def Vid(self) -> float:
        if self.I<0:
            Vid_val = self.Wid*1e3/(-self.I*self.EffFar*self.Ncells)
        else: 
            Vid_val = self.Wid*1e3/(-self.I/self.EffFar*self.Ncells)
        return Vid_val

    @property # Note that I and j are no longer the same concept. I is the external circuit electronic current, and j is the internal cell ionic current
    def j(self) -> float:
        if self.I<0:
            j_val = self.I*self.EffFar/self.cellArea
        else: 
            j_val = self.I/self.EffFar/self.cellArea
        return j_val        

    @j.setter
    def j(self, val:float):
        if val < 0:
            self.I = val*self.cellArea/self.EffFar
        else:
            self.I = val*self.cellArea*self.EffFar

    @property
    def Vop(self) -> float:
        return self.Vid - self.f_ASR*self.ASR()*self.j

    @property
    def W(self) -> float:
        return self.Vop*(-self.I)*self.Ncells*1e-3
        
    @property
    def Q(self) -> float:
        return self.F1.n*self.F1.h + self.A1.n*self.A1.h - self.F0.n*self.F0.h - self.A0.n*self.A0.h - self.W - self.Qloss 
    
    @property
    def Exth(self) -> float:
        return self.Q*(1 - (298.15/self.F1.T))  
    
    @property
    @override
    def Exloss(self) -> float:
        return self.F0.Ex*self.F0.n + self.A0.Ex*self.A0.n - self.F1.Ex*self.F1.n - self.A1.Ex*self.A1.n + self.W + self.Exth 
    
    @property
    def RC(self) -> float:
        if self.I<0:
            val = -self.Ncells*self.I*self.EffFar/(96485*self.F0.n*2*self.F0.Xi[4]) 
        else:
            val = self.Ncells*self.I/self.EffFar/(2*96485*self.F0.n*(self.F0.Xi[0] + 4*self.F0.Xi[1] + self.F0.Xi[3]))
        return val
    # Add a warning or error if steam-carbon (or O:C) ratio at inlet is less than 2.
    # Consider defining these in terms of inlet and outlet species rather than current??
    
    # @property
    # def RC_EC(self):
    #     return (self.F0.n*(self.F0.Xi[2] + self.F0.Xi[4]) - self.F1.n*(self.F1.Xi[2] + self.F1.Xi[4]))/(
    #             self.F0.n*(self.F0.Xi[2] + self.F0.Xi[4]))

    def K_WGS(self) -> float:
        return np.exp(-(32.1153*(self.F1.T) - 3.5211e4)/(8.314*self.F1.T)) 
    
    def K_MSR(self) -> float:
        return np.exp(-(225215.699 - 252.643*self.F1.T)/(8.314*self.F1.T)) 

    def ASR(self, asr_choice:Literal[0,1,2,3,4]=1) -> float:
        '''
        Choice of ASR function to use. Choose an integer from 0 to 4.
            0 - Use only for calibration. ASR=1, and then the effective ASR value is entirely decided by the variable factor f_ASR
            1 - (default) Polynomial fit from data of Braun et al. 2019, DOI 10.1149/09101.0997ecst. ===> Accurate, but valid STRICTLY ONLY from 500C/773K to 650C/923K
            2 - Exponential fit from data of Braun et al. 2019, DOI 10.1149/09101.0997ecst ===> Less accurate, but should be valid at a wider temperature range.
            3 - Constant from H. An, Nature Energy 2018,  DOI 10.1038/s41560-018-0230-0
            4 - Abritrarily fixed at 10% of the SOC ASR function from M. Riedel 2020 DOI 10.1149/1945-7111/ab6820
        '''
        match asr_choice:
            case 0: # Value used during calibration, so that the ASR is fully decided by f_ASR
                _ASR = 1 # ohm.cm2
            case 1: # Polynomial fit from data of Braun et al. 2019, DOI 10.1149/09101.0997ecst ===> USE ONLY FROM 500C/773K to 650C/923K
                _ASR = 2.06006613e+02 - self.F1.T*6.78960141e-01 + (self.F1.T**2)*7.49630871e-04 - (self.F1.T**3)*2.76745905e-07 # ohm.cm2
                if self.Tpcc1 < 770 or self.Tpcc1 > 925:
                    raise ValueError (f'PCC temperature ({self.Tpcc1}) is outside validity range of ASR fitted function (773 to 923 K).')
            case 2: # Exponential fit from data of Braun et al. 2019, DOI 10.1149/09101.0997ecst ===> Less accurate, but should be valid at a wider temperature range.
                _ASR = 9.21396991e2 * np.exp(-8.63749532e-3 * self.F1.T) # ohm.cm2
            case 3: # H. An, Nature Energy 2018,  DOI 10.1038/s41560-018-0230-0
                _ASR = 0.09 # ohm.cm2
            case 4: # M. Riedel 2020 DOI 10.1149/1945-7111/ab6820 # ASR reduced by a factor of 10 to account for lower PCC temperature, and avoid very low voltage in FC mode.
                _ASR = 7.28e2*np.exp(-8.29e-3*((self.F1.T)-273.15))/10  # ohm.cm2
            case _:
                raise ValueError('ASR choice is wrong')
        return _ASR

    def cform(self, side:str='out'):
        if side == 'out':
            f = self.F1
        else:
            f = self.F0
        return funs.carbonFraction(f.T, f.P, dict(zip(f.Medium.species, f.Xi)))

    @property
    def Sgen(self) -> float:
        return self.F1.s*self.F1.n + self.A1.s*self.A1.n - self.F0.s*self.F0.n - self.A0.s*self.A0.n - self.Q/self.F1.T

    @property
    def Lambda(self) -> float:
        if self.I > 0:
            val:float = (self.A0.n*self.A0.Xi[0])/(self.n_H*4)
        else:
            val = np.nan
        return val

    def f_sol(self, x:NDArray[np.float64]):
        x = np.abs(x)
        self.F1.Xi =  np.concatenate((x[:5], [1-np.sum(x[:5])]))
        self.F1.n = x[5]

        return self.Eqs()
        
    def run_sol(self, x0:NDArray[np.float64]|None = None):
        if x0 is None:
            x0 = np.concatenate((self.F0.Xi[:5], [self.F0.n]))

        lower_bounds = [1e-8,1e-8,1e-8,1e-8,1e-8,1e-8*self.Ncells]
        upper_bounds = [0.9999,0.9999,0.9999,0.9999,0.9999,1e1*self.Ncells]
        x0_clipped = np.clip(np.abs(x0), lower_bounds, upper_bounds)

        sol = spo.least_squares(self.f_sol, x0_clipped, xtol=1e-15, bounds = (lower_bounds, upper_bounds)) 
        if np.linalg.norm(sol.fun)<=1e-7:
            sol = spo.root(self.f_sol, sol.x)
        if np.linalg.norm(sol.fun)>1e-7:
            sol = spo.root(self.f_sol, x0)
        res = self.f_sol(sol.x)
        # Failure criteria changed to be a mix of relative and absolute.
            # sol.success is relative, and can fails at small currents.
            # the residuals are absolute, and can fail at large currents.
            # So the solver should fail only when both checks fail together.
        if not sol.success and np.linalg.norm(res)>1e-10:  
            print(f'{self.name} failed, residuals = {res}, norm = {np.linalg.norm(res)}')
        self._sol = sol
        return sol



class PCC_delG(PCC_base):
    '''
    PCC model that uses overall Gibbs energy change to calculate ideal voltage
    '''
    @property
    @override
    def Wid(self) -> float:
        return (self.F1.n*self.F1.g + self.A1.n*self.A1.g - self.F0.n*self.F0.g - self.A0.n*self.A0.g) - self.Wchem

class PCC_Nernst_avg(PCC_base):
    '''
    PCC model that uses Nernst equation with average flow compositions to calculate ideal voltage
    '''
    @property
    def Vnernst(self) -> float:
        R = 8.314462618
        E0: float = ((self.A1.Medium.g_i[2] - self.F1.Medium.g_i[0]- 0.5*self.A1.Medium.g_i[0])*1e3/2/96485) 
        aver = np.mean([[self.F0.Xi[0], self.F1.Xi[0]], [self.A0.Xi[2], self.A1.Xi[2]], [self.A0.Xi[0], self.A1.Xi[0]]], axis=1)
        
        _x_avg_names = ["H2" ,"H2O", "O2"]
        for index in np.where(aver < 1e-6)[0]:
            print(f"Mole fraction {_x_avg_names[index]} is (almost) 0")  
        
        Eid: float = -E0 - R*self.A1.T/2/96485*(np.log(aver[1])-np.log(aver[0])-0.5*np.log(self.P_pcc*aver[2])) 
        
        return Eid

    @property
    @override
    def Wid(self) -> float:
        return (-self.Vnernst*self.j*self.cellArea*self.Ncells/1e3) - self.Wchem


class AEL(ModelBase):
    '''
    Alkaline electrolyser model
    It determines the current density [A/cm2] (j) and outlet mole flow and composition for a given mole flow of water
    
    '''
    def __init__(self, name:str, Medium, T:float=298.15, P:float=1, j:float=-0.32, Ncells:int|None=None, A:float|None=None,
                 F0_n:float=1, F0_Xi:NDArray[np.float64]=np.array([1, 0, 0, 0]), 
                 F1_n:float=1, F1_Xi:NDArray[np.float64]|None=None, #['H2O(L)', 'H2', 'O2', 'H2O']
                 F2_n:float=1, F2_Xi:NDArray[np.float64]=np.array([0, 0, 1, 0]) 
                ): 
                
        '''
        media.Media_AEL()
        '''
        super().__init__(locals())
        
        self.F0: interfaces.Flange = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P)
        self.F1: interfaces.Flange = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P) #H2
        self.F2: interfaces.Flange = interfaces.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=T, P=P) #O2
        self._j: float = j
        self.A = A
        self.Ncells: int|None = Ncells

    @override    
    def Eqs(self):
        y = np.array((
            [self.F0.Xi[0]+self.F0.Xi[3], self.F2.Xi[0]+self.F2.Xi[3], self.F1.Xi[0]+self.F1.Xi[3], 1], # H2O
            [self.F0.Xi[1], self.F2.Xi[1], self.F1.Xi[1], -1], # H2
            [self.F0.Xi[2], self.F2.Xi[2], self.F1.Xi[2], -0.5])) # O2
        
        n = np.array([self.F0.n, - self.F2.n, -self.F1.n, self.Ncells*self.I/2/96485])

        self.Eq: NDArray[np.float64] = np.concatenate((
            [
            np.sum(self.F1.Xi) - 1,
            ],
            np.dot(y, n),
        ))
            
        self.F1.P = self.F0.P
        self.F2.P = self.F0.P  
        
        return self.Eq

    @property
    def j(self):
        return self._j
    
    @j.setter
    def j(self, val:float):
        self._j = val
    
    @property
    def I(self) -> float:
        return self.j*self.A
    
    @property
    def Vop(self):
        return 1.481 + 2.9*1e-3*self.F0.P - self.j*1.05

    @property
    def W(self) -> float:
        return -self.I*self.Vop*self.Ncells*1e-3 # to kJ/s

    @property
    def Q(self):
        return self.F2.n*self.F2.h + self.F1.n*self.F1.h - self.F0.n*self.F0.h - self.W
    
    @property
    def Exth(self):
        return self.Q*(1 - 298.15/self.F1.T)
    
    @property
    @override
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex + self.Exth + self.W
    
    def MBcheck(self):
        print(f'{abs((self.F0.m/(self.F1.m + self.F2.m) -1))*100 }% difference')
        return self.F0.m - (self.F1.m + self.F2.m)

    @property
    def Exeff(self):
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.max([0, self.Exth]) +  self.W)
        
    def f_sol(self, x:NDArray[np.float64]) -> NDArray[np.float64]:
        self.j = x[0]
        self.F1.n = x[1]
        self.F1.Xi = np.array([0, x[2], 0.001, 0.002])
        self.F2.n = x[3]
        return self.Eqs()
        
    def run_sol(self):
        x0 = np.array([self.j,self.F0.n,0.997,self.F0.n])  
        sol = spo.least_squares(self.f_sol, x0)
        return sol
