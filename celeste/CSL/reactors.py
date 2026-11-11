"""
Reactors models

1) EquilBase: Base model for isothermal equilibirum reactors
2) MR: Methanation/RWGS
3) HB: Haber Bosch
4) MeOH_Equil
...


"""

import numpy as np
from numpy.typing import NDArray as npArray
import scipy.optimize as spo

from . import ModelBase, SysBase, interfaces, media


class EquilBase(ModelBase):
    """
    
    """

    def __init__(self, name:str, Medium:media.Medium_Base, 
                 T:float=298.15, P:float=1.0, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None, #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None,
                 ):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P)
        self._P = P
        self._T = T

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P_):
        self.__dict__['P'] = P_
        self._P = P_
        self.F0.P = P_
        self.F1.P = P_   

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, T_):
        self.__dict__['T'] = T_
        self._T = T_
        self.F0.T = T_
        self.F1.T = T_   

    def Z(self):
        return ((1000/self.F1.T)-1)    

class MR(EquilBase):
    """ 
        Medium: media.Fuel_CH4()
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_CH4(), 
                 T:float=500.15, P:float=23.0, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None, #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None
                ):          
        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q) 
        self._N_H = np.array([2,4,0,0,2,0])
        self._N_C = np.array([0,1,1,1,0,0])
        self._N_O = np.array([0,0,2,1,1,0])
        self._N_N = np.array([0,0,0,0,0,2])
        
    def Eqs(self):
        
        self.Eq:npArray[np.float64] = np.array([
            # MB
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)), 
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)), 
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)), 
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)), 
            np.sum(self.F1.Xi) - 1, 
            # Equil
            self.F1.Xi[3]*self.F1.Xi[4]*self.K_WGS() - (self.F1.Xi[2]*self.F1.Xi[0]), 
            self.F1.Xi[1]*self.F1.Xi[4]*self.K_MSR() - (self.F1.Xi[3]*self.F1.Xi[0]**3*self.F1.P**2), 
        ])
        self.Q = (self.F1.n*self.F1.h - self.F0.n*self.F0.h )
        
        self.F0_n = self.F0.n
        self.F0_Xi = self.F0.Xi
        self.F1_n = self.F1.n
        self.F1_Xi = self.F1.Xi
        
        return self.Eq
    
    def K_WGS(self):
        #return (np.exp((-0.2935*self.Z()**3) + (0.6351*self.Z()**2) + (4.1788*self.Z()) + 0.3169))
        return np.exp(- (32.1153*(self.F1.T) - 3.5211e4)/(8.314*self.F1.T)) 
    
    def K_MSR(self):
        #return (1.0267*np.exp((-0.2513*self.Z()**4) + (0.3665*self.Z()**3) + (0.581*self.Z()**2) - 27.134*self.Z() + 3.277))
        return np.exp(-(225215.699 -252.643*self.F1.T)/(8.314*self.F1.T)) 
    
    def MBcheck(self)->float:
        # Mass balance
        MB:float = self.F0.m - self.F1.m 
        print(f"{abs((self.F0.m/(self.F1.m)) - 1)*100}% difference ")    
        return MB
    
    @property
    def Exth(self):
        return self.Q*(1 - (298.15/self.F1.T))

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth
    
    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n - self.Q/self.F0.T
        
    def f_sol(self, x):  
        self.F1.n = x[0]
        self.F1.Xi = x[1:7] 
        return self.Eqs()
        
    def run_sol(self, method=spo.least_squares):  
        x0 = np.concatenate(([self.F0.n], self.F0.Xi)) 
        sol = method(self.f_sol, x0)
        self.f_sol(sol.x)
        return sol

class HB(EquilBase):
    """
    Isothermal and isobaric Haber-Bosch reactor with two different kinetic models
    
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.HB(), AF=1.0,
                 T:float=673.15, P:float=200.0, Q:float=0.0, Vol:float=1.0, r_type:bool=True,
                 F0_n:float=1.0,  F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0,  F1_Xi:npArray[np.float64]|None=None,
                ):  
        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q)
        
        self.AF = AF # catalyst activity factor. Morud, 2004 wrote that now this could be a good number
        self.Vol = Vol # reactor volume
        self.r_type = r_type
        #self.Q = Q

        if self.r_type:
            self.r = self.r_Nielsen
        else:
            self.r = self.r_TP
        
    def Eqs(self):
        
        self.F1.P = self.F0.P
        self.F1.T = self.F0.T 
        self.tau0:float = self.F0.P*self.Vol*1e3/0.08314/self.F0.T/self.F0.n # s
        self.tau1:float = self.F1.P*self.Vol*1e3/0.08314/self.F1.T/self.F1.n # s       

        self.Eq:npArray[np.float64] = np.array([
            np.sum(self.F1.Xi) - 1,
            self.F0.Xi[0]*self.F0.n - self.F1.Xi[0]*self.F1.n - (3/2)*self.r(self.F1.T, self.F1.P, self.F1.Xi, self.AF)*self.Vol,
            self.F0.Xi[1]*self.F0.n - self.F1.Xi[1]*self.F1.n - (1/2)*self.r(self.F1.T, self.F1.P, self.F1.Xi, self.AF)*self.Vol,
            self.F0.Xi[2]*self.F0.n - self.F1.Xi[2]*self.F1.n + self.r(self.F1.T, self.F1.P, self.F1.Xi, self.AF)*self.Vol,
        ])
        
        self.Q = self.F1.n*self.F1.h - self.F0.n*self.F0.h
               
        self.F0_n = self.F0.n
        self.F0_Xi = self.F0.Xi
        self.F1_n = self.F1.n
        self.F1_Xi = self.F1.Xi
        
        return self.Eq
    
    @staticmethod
    def r_Nielsen(T, P, Xi, AF):
        """
        Kinetic from Aspen Plus, 2015, "Aspen Plus ammonia model", page 16-17 and from, Nielsen, 1964, "Ammonia synthesis rate and mechanism".
        Another adjustment was made from the book A. Nielsen, 1995, "Ammonia catalysis and manufacture", page 169.
        first commented values: from Aspen plus. Second commented values: Nielsen with lowest error.
        Considers: 
        - an iron catalyst
        - activities of H2, N2 and NH3
        """
        R = 1.987 # ideal gas const cal/mol/K
        alpha = 0.64 # 0.654 
        w = 1.564 # 1.523 
        Ru = 0.08206 # atm*L/mol/K
        Patm_bar = 1/1.01325 #atm/bar
        A = np.array([0.1975, 1.3445, 2.393])
        B = np.array([0.02096, 0.05046, 0.03415])
        C = np.array([5.04e2, 4.2e4, 4.7687e6])

        K_a = 10**(-2.691122*np.log10(T) - 5.519265e-5*T + 1.848863e-7*T**2 + 2001.6/T + 2.6899)
        k_2 =  1.06e13*np.exp(-17429/T/R) # in Nielsen, 1964 the values is 2.12*10e13, but a correction is mentioned in the book Nielsen, 1995 page 169
        K_3 = 3.07e-2*np.exp(19361/T/R)  # E in cal/mol
        # pressure in atm, Ru in atm m3/kmol/K, T in K, a in atm*Xi
        a = Xi*P*Patm_bar*np.exp((P*Patm_bar/Ru/T)*(B - A/Ru/T - C/(T**3) + ((A**0.5 - np.dot(Xi, A**0.5))**2 )/Ru/T) )

        r = 1e3*AF*(k_2*(a[1]*K_a**2 - (a[2]**2)/(a[0]**3) )/((1 + K_3*(a[2]/(a[0]**w)) )**(2*alpha)) # mol/m3/hr --> mol of NH3 per m3 of catalyst bed per hour
             )/3600 # mol NH3/m3_catalyst_bed/hr to second
        return r
    
    @staticmethod
    def r_TP(T, P, Xi, AF):
        """
        General Temkin-Pyzhev rate equation from Morud, 1998, "Analysis of instability in an industrial ammonia reactor,
        and adjusted from Nielsen, 1964, "Ammonia synthesis rate and mechanism".

        """
        R = 8.314
        alpha = 0.5
        betha = 0.5
        rho = 2200 # kg_catalyser/m3 from Morud and skogestad. In case r needs to be in mol NH3/kg_catalyser/h
        k1 = 1.79e4*np.exp(-87090/R/T)
        k2 = 2.57e16*np.exp(-198464/R/T)
        r = 2*AF*1e3*(k1*Xi[1]*P*((Xi[0]*P)**3/(Xi[2]*P)**2)**alpha -
                           k2*((Xi[2]*P)**2/(Xi[0]*P)**3)**betha)/3600 # mol NH3/m3/hr to second
        return r

    def MBcheck(self)->float:
        # Mass balance
        MB:float = self.F0.m - self.F1.m
        print(f" Syngas ratio: {(self.F0.Xi[0]/self.F0.Xi[1])}")
        print(f"{abs((self.F0.m/(self.F1.m)) - 1)*100}% difference ")    
        return MB
        
    @property
    def Exth(self):
        return self.Q*(1 - (298.15/self.F1.T))

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth
    
    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n - self.Q/self.F0.T
        
    def for_sol(self, x):
        self.F1.n = x[0]
        self.F1.Xi = x[1:4]
        return self.Eqs()    
    
    def run_sol(self):
        x0 = np.concatenate(([self.F0.n], self.F0.Xi))
        sol = spo.least_squares(self.for_sol, x0) 
        self.for_sol(sol.x)
        return sol

class HB_Adiabatic(ModelBase):
    """
    Adiabatic Haber-Bosch reactor with two different kinetic models.
    The inlet and outlet pressure can be different.
    
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.HB(), 
                 AF=1.0, Vol:float=1.0, r_type:bool=True, P:float=200.0,
                 F0_n:float=1.0,  F0_Xi:npArray[np.float64]|None=None, F0_T:float=673.15, F0_P:float=200.0,
                 F1_n:float=1.0,  F1_Xi:npArray[np.float64]|None=None, F1_T:float=673.15, F1_P:float=200.0
                ):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P)
 
        self.AF = AF # catalyst activity factor. Morud, 2004 wrote that now 4.75 this could be a good number
        self.Vol = Vol # reactor volume
        self.r_type = r_type
        if self.r_type:
            self.r = self.r_Nielsen
        else:
            self.r = self.r_TP
        
    def Eqs(self):
        self.tau0:float = self.F0.P*self.Vol*1e3/0.08314/self.F0.T/self.F0.n # s 
        self.tau1:float = self.F1.P*self.Vol*1e3/0.08314/self.F1.T/self.F1.n # s 

        r = self.r(self.F1.T, self.F1.P, self.F1.Xi, self.AF)*self.Vol
        self.Eq:npArray[np.float64] = np.array([
            np.sum(self.F1.Xi) - 1,
            self.F0.Xi[0]*self.F0.n - self.F1.Xi[0]*self.F1.n - (3/2)*r,
            self.F0.Xi[1]*self.F0.n - self.F1.Xi[1]*self.F1.n - (1/2)*r,
            self.F0.Xi[2]*self.F0.n - self.F1.Xi[2]*self.F1.n + r,
            self.F0.n*self.F0.h - self.F1.n*self.F1.h,
        ])

        return self.Eq
    
    @staticmethod
    def r_Nielsen(T, P, Xi, AF):
        """
        Kinetic from Aspen Plus, 2015, "Aspen Plus ammonia model", page 16-17 and from, Nielsen, 1964, "Ammonia synthesis rate and mechanism".
        Another adjustment can be made from the book A. Nielsen, 1995, "Ammonia catalysis and manufacture", page 169.
        first commented values: from Aspen plus. Second commented values: Nielsen with lowest error.
        Considers: 
        - an iron catalyst
        - activities of H2, N2 and NH3
        """
        R = 1.987 # ideal gas const cal/mol/K
        alpha =  0.654 #0.64 # 0.654 
        w = 1.523 # 1.564 # 1.523 
        Ru = 0.08206 # atm*L/mol/K
        Patm_bar = 1/1.01325 #atm/bar
        A = np.array([0.1975, 1.3445, 2.393])
        B = np.array([0.02096, 0.05046, 0.03415])
        C = np.array([5.04e2, 4.2e4, 4.7687e6])

        k_2 = 3.945e10*np.exp(-11171/T/R) #1.06e13*np.exp(-17429/T/R) # in Nielsen, 1964 the values is 2.12*10e13, but a correction is mentioned in the book Nielsen, 1995 page 169
        K_3 = 2.94e-4*np.exp(24052/T/R) #3.07e-2*np.exp(19361/T/R)  # E in cal/mol
        # pressure in atm, Ru in atm*L/mol/K, T in K, a in atm*Xi
        a = Xi*P*Patm_bar*np.exp((P*Patm_bar/Ru/T)*(B - A/Ru/T - C/(T**3) + ((A**0.5 - np.dot(Xi, A**0.5))**2 )/Ru/T) )
        if Xi[2] < 1e-5 or Xi[0] < 1e-3:
            r = float('NaN')
        else:
            K_a = 10**(-2.691122*np.log10(T) - 5.519265e-5*T + 1.848863e-7*T**2 + 2001.6/T + 2.6899)
            r = 1e3*AF*k_2*((a[1]*K_a**2 - (a[2]**2)/(a[0]**3) )/((1. + K_3*a[2]/(a[0]**w))**(2*alpha)) # mol/m3/hr --> mol of NH3 per m3 per hour
                )/3600 # mol NH3/m3_catalyst_bed/hr to second
        return r
    
    @staticmethod
    def r_TP(T, P, Xi, AF):
        """
        General Temkin-Pyzhev rate equation from Morud, 1998, "Analysis of instability in an industrial ammonia reactor,
        and adjusted from Nielsen, 1964, "Ammonia synthesis rate and mechanism".
        rho: is the bulk density (rho_bulk) kg_catalyser/m3 from Morud and skogestad. In case r needs to be in mol NH3/kg_catalyser/h.
            The bulk density considers the void volume, therefore the r is in terms of reactor volume.

        """
        R = 8.314
        alpha = 0.5
        rho = 2200 # kg_catalyser/m3 from Morud and skogestad. In case r needs to be in mol NH3/kg_catalyser/h
        if Xi[2] < 1e-5 or Xi[0] < 1e-3:
            r = float('NaN')
        else:
            k1 = 1.79e4*np.exp(-87090/R/T)
            k2 = 2.57e16*np.exp(-198464/R/T)
            r = 2*AF*1e3*(k1*Xi[1]*P*((Xi[0]*P)**3/(Xi[2]*P)**2)**alpha -
                               k2*((Xi[2]*P)**2/(Xi[0]*P)**3)**(1-alpha))/3600 # mol NH3/m3 reactor/hr to second
        return r

    def MBcheck(self)->float:
        # Mass balance
        MB:float = self.F0.m - self.F1.m
        print(f" Syngas ratio: {(self.F0.Xi[0]/self.F0.Xi[1])}")
        print(f"{abs((self.F0.m/(self.F1.m)) - 1)*100}% difference ")    
        return MB
        
    @property
    def Exth(self):
        return 0 #self.Q*(1 - (298.15/self.F1.T))

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth
    
    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n 

    @property
    def conversion(self):
        return (self.F0.Xi[0]*self.F0.n - self.F1.Xi[0]*self.F1.n)/(self.F0.Xi[0]*self.F0.n)
        
    def for_sol(self, x):
        self.F1.n = np.abs(x[0])
        self.F1.Xi = np.abs(x[1:4])
        self.F1.T = np.abs(x[4])
        return self.Eqs()    
    
    def run_sol(self):
        x0=np.concatenate(([self.F0.n*0.75], np.array([0.57, 0.16, 0.27]), [self.F0.T*1.2])) # maybe make this more flexible 
        sol = spo.root(self.for_sol, x0, method='hybr') 
        self.for_sol(sol.x)
        if not sol.success:
            print(f'{self.name} Failed')

        return sol

class MeOH_Equil(EquilBase):
    """
    Equilibrium isothermal methanol synthesis reactor.
    Two reactions are considered to reach equilibrium:
        1) CO2 + 3H2 <-> CH3OH + H2O
        2) CO2 + H2 <-> CO + H2O
    
    Parameters:
        name: object's name
        Medium: media of the inlet stream. Defaults to media.Fuel_MeOH()
        T: temperature in [K]. Defaults to 473.15 
        P: pressure in [bar]. Defaults to 50.0
        Q: heat released/absorbed in [kJ/s]. Defaults to 0.0 
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium1. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium2. Defaults to None   
    
    Attributes: 
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        Exth: thermal exergy of the object in [kJ/s]
        Exloss: exergy loss in the object in [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Notes:
        Reference: Graaf 1985 "Chemical Equilibria in Methanol Synthesis"
    
    Examples:
        MeOH_Equilibrium = MeOH_Equil.Reactors('MeOH_example', Medium=media.Fuel_MeOH(), F0_n=100, 
        F0_Xi=np.array([0.75, 0.05, 0.1, 0.1, 1e-8, 1e-8, 1e-8]), T=473.15, P=50.0)
         
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_MeOH(), 
                 T:float=473.15, P:float=50.0, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None, #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None,
                ):
        
        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q)   
        self._N_H = np.array([2,4,0,0,2,0,4])
        self._N_C = np.array([0,1,1,1,0,0,1])
        self._N_O = np.array([0,0,2,1,1,0,1])
        self._N_N = np.array([0,0,0,0,0,2,0])
        
    def Eqs(self):

        self.F1.P = self.F0.P
        self.F1.T = self.F0.T 

        self.Eq:npArray[np.float64] = np.array([
            # MB
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)), # C balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)), # H balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)), # O balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)), # N balance
            self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1], # CH4
            1 - np.sum(self.F1.Xi), 
            # Equil
            self.F1.Xi[0]*self.F1.Xi[2]*self.K_rWGS() - (self.F1.Xi[3]*self.F1.Xi[4]), 
            self.F1.Xi[0]**3*self.F1.Xi[2]*self.P**2*self.K_MeOH() - (self.F1.Xi[6]*self.F1.Xi[4]), # for CO2
            #(self.F1.Xi[0]**2*self.F1.Xi[3]*self.P**2*self.K_MeOH()) - self.F1.Xi[6] # for CO   
        ])

        self.Q = (self.F1.n*self.F1.h - self.F0.n*self.F0.h )  

        self.F0_n = self.F0.n
        self.F0_Xi = self.F0.Xi
        self.F1_n = self.F1.n
        self.F1_Xi = self.F1.Xi

        return self.Eq
    
    def K_rWGS(self):
        return 10**((-2073/self.F1.T) + 2.029)
    
    def K_MeOH(self):
        return 10**((3066/self.F1.T) - 10.592) # for CO2
        #return 10**((5139/self.F1.T) - 12.621) # for CO 
    
    def MBcheck(self)->float:
        self.m0:float = self.F0.n*self.F0.mw
        self.m1:float = self.F1.n*self.F1.mw
        MB:float = self.m1 - self.m0
        print(f"{(self.F1.n*self.F1.mw/(self.F0.n*self.F0.mw) - 1)*100}% difference")
        return MB
    
    def ExBs(self)->None:
        self.F0.ExBs()
        self.F1.ExBs()
        self.Exth:float = self.Q*(1 - (298.15/self.F1.T))
        self.Exloss:float = self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth
        self.Exeff:float = 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.maximum(0, self.Exth))
    
    def f_sol(self, x):
        self.F1.n = x[0]
        self.F1.Xi = x[1:8]
        return self.Eqs()
        
    def run_sol(self):
        x0 = np.concatenate(([self.F0.n], media.Fuel_MeOH()._reference_X )) 
        sol = spo.least_squares(self.f_sol, x0)
        self.f_sol(sol.x)
        return sol

class MeOH_Yield(EquilBase):
    """
    Methanol synthesis isothermal reactor where the methanol yield is given and the rWGS reaction is in equilibrium 
    Two reactions are considered:
        1) CO2 + 3H2 <-> CH3OH + H2O (yield)
        2) CO2 + H2 <-> CO + H2O (equilibrium) 
    
    Parameters:
        name: object's name
        Medium: media of the inlet stream. Defaults to media.Fuel_MeOH()
        Yield: methanol yield. Its goes from 0 to 1. Defaults to 0.35
        T: temperature in [K]. Defaults to 473.15 
        P: pressure in [bar]. Defaults to 50.0
        Q: heat released/absorbed in [kJ/s]. Defaults to 0.0 
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium1. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium2. Defaults to None   
    
    Attributes: 
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        Exth: thermal exergy of the object in [kJ/s]
        Exloss: exergy loss in the object in [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Notes:
        Reference: Graaf 1985 "Chemical Equilibria in Methanol Synthesis"
    
    Examples:
        MeOH_Yield = MeOH_yield.Reactors('MeOH_example', Medium=media.Fuel_MeOH(), Yield=0.33, F0_n=100, 
        F0_Xi=np.array([0.75, 0.05, 0.1, 0.1, 1e-8, 1e-8, 1e-8]), T=473.15, P=50.0)
    
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_MeOH(), 
                 Yield:float=0.35, T:float=473.15, P:float=50.0, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None,  #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None
                 ):
        
        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q)   
        self._N_H = np.array([2,4,0,0,2,0,4])
        self._N_C = np.array([0,1,1,1,0,0,1])
        self._N_O = np.array([0,0,2,1,1,0,1])
        self._N_N = np.array([0,0,0,0,0,2,0])
        self.Yield = Yield
        
    def Eqs(self):

        self.F1.P = self.F0.P
        self.F1.T = self.F0.T
        
        self.Eq:npArray[np.float64] = np.array([
            # MB
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)), # C balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)), # H balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)), # O balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)), # N balance
            self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1], # CH4
            1 - np.sum(self.F1.Xi), # Xi balance
            (self.Yield*self.F0.n*(self.F0.Xi[2]+self.F0.Xi[3]) + self.F0.n*self.F0.Xi[6]) - (self.F1.n*self.F1.Xi[6]), #MeOH            
            # Equil
            self.F1.Xi[0]*self.F1.Xi[2]*self.K_rWGS() - (self.F1.Xi[3]*self.F1.Xi[4]), 
        ])

        self.Q = (self.F1.n*self.F1.h - self.F0.n*self.F0.h )

        self.F0_n = self.F0.n
        self.F0_Xi = self.F0.Xi
        self.F1_n = self.F1.n
        self.F1_Xi = self.F1.Xi
        
        return self.Eq
    
    def K_rWGS(self):
        return 10**((-2073/self.T) + 2.029)
    
    def MBcheck(self)->float:
        self.m0:float = self.F0.n*self.F0.mw
        self.m1:float = self.F1.n*self.F1.mw
        MB:float = self.m1 - self.m0
        print(f"{(self.F1.n*self.F1.mw/(self.F0.n*self.F0.mw) - 1)*100}% difference")
        return MB
    
    def ExBs(self)->None:
        self.Exth:float = self.Q*(1 - (298.15/self.F1.T))
        self.Exloss:float = self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth
        self.Exeff:float = 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.maximum(0, self.Exth))
    
    def f_sol(self, x): 
        self.F1.n = x[0]
        self.F1.Xi = x[1:8]
        return self.Eqs()
        
    def run_sol(self):
        x0 = np.concatenate(([self.F0.n], media.Fuel_MeOH()._reference_X )) 
        sol = spo.least_squares(self.f_sol, x0)
        self.f_sol(sol.x)
        return sol

class MeOH_Kinetic(EquilBase):
    """
    Isothermal kinetic based reactor model for methanol synthesis
    Two reactions are considered:
        1) CO2 + 3H2 <-> CH3OH + H2O 
        2) CO2 + H2 <-> CO + H2O 
    
    Parameters:
        name: object's name
        Medium: media of the inlet stream. Defaults to media.Fuel_MeOH()
        M_cat: mass of the catalyst in [kg]. Defaults to None
        T: temperature in [K]. Defaults to 473.15 
        P: pressure in [bar]. Defaults to 50.0
        Q: heat released/absorbed in [kJ/s]. Defaults to 0.0 
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium1. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium2. Defaults to None   
    
    Attributes: 
        R: ideal gas constant in [J/mol]
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        Exth: thermal exergy of the object in [kJ/s]
        Exloss: exergy loss in the object in [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        r_MeOH: determination of reaction rate of the CO2 to methanol reaction (CO2-Hydrogenation). Returns: float in [mol/kg_cat*s]
        r_rWGS: determination of reaction rate of the reverse water gas shift reaction (rWGS). Returns: float in [mol/kg_cat*s]
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Notes:
        Reference: Van den Bussche and Froment, 1996, "A steady-state kinetic model for methanol synthesis and the water 
        gas shift reaction in a commercial catalyst"
       
    Examples:
        MeOH_Kinetic = MeOH_Kinetic.Reactors('MeOH_example', Medium=media.Fuel_MeOH(), M_cat=1000, F0_n=100, 
        F0_Xi=np.array([0.75, 0.05, 0.1, 0.1, 1e-8, 1e-8, 1e-8]), T=473.15, P=50.0)

    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_MeOH(), M_cat:float=None,
                 T:float=473.15, P:float=50, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None,  #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None
                 ):

        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q)   
        self.M_cat = M_cat
        self.R:float = 8.314462618 # J/mol

        self.F1.P = self.F0.P
        self.F1.T = self.F0.T   
        # adsorption constants
        self._K_a = 3453.38
        self._K_b = 0.499*np.exp(17197/(self.R*self.F1.T)) 
        self._K_c = 6.62e-11*np.exp(124119/(self.R*self.F1.T))

    def Eqs(self): 
        rrWGS = self.r_rWGS()   
        rMeOH = self.r_MeOH()
        self.Eq:npArray[np.float64] = np.array([          
            # MB
            (self.F0.n*self.F0.Xi[0] - 3*rMeOH*self.M_cat - rrWGS*self.M_cat) - self.F1.n*self.F1.Xi[0], # H2 balance
            (self.F0.n*self.F0.Xi[1]) - self.F1.n*self.F1.Xi[1], # CH4 balance 
            (self.F0.n*self.F0.Xi[2] - rMeOH*self.M_cat - rrWGS*self.M_cat) - self.F1.n*self.F1.Xi[2], # CO2 balance 
            (self.F0.n*self.F0.Xi[3] + rrWGS*self.M_cat) - self.F1.n*self.F1.Xi[3], # CO balance            
            (self.F0.n*self.F0.Xi[4] + rMeOH*self.M_cat + rrWGS*self.M_cat) - self.F1.n*self.F1.Xi[4], # H2O balance
            (self.F0.n*self.F0.Xi[5]) - self.F1.n*self.F1.Xi[5], # N2 balance
            (self.F0.n*self.F0.Xi[6] + rMeOH*self.M_cat) - self.F1.n*self.F1.Xi[6], # CH3OH balance                                
            1 - np.sum(self.F1.Xi), # Xi balance
        ])
        
        self.Q = (self.F1.n*self.F1.h - self.F0.n*self.F0.h )

        self.F0_n = self.F0.n
        self.F0_Xi = self.F0.Xi
        self.F1_n = self.F1.n
        self.F1_Xi = self.F1.Xi

        return self.Eq

    def r_MeOH(self)->float: 
        # Equilibrium constant
        K_meoh = 10**((3066/self.F1.T) - 10.592)
        # kinetic factor
        k_meoh = 1.07*np.exp(36696/(self.R*self.F1.T)) #mol/kg_cat*s*bar^2
        r_num = k_meoh*self.F1.Xi[0]*self.F1.Xi[2]*self.F1.P**2*(1 - (1/K_meoh)*self.F1.Xi[4]*self.F1.Xi[6]/(self.F1.Xi[0]**3*self.F1.Xi[2]*self.F1.P**2))
        r_den = (1 + self._K_a*(self.F1.Xi[4]/self.F1.Xi[0]) + self._K_b*(self.F1.Xi[0]*self.F1.P)**0.5 + self._K_c*self.F1.Xi[4]*self.F1.P)**3 
        return r_num/r_den 
    
    def r_rWGS(self)->float: 
        # Equilibrium constant
        K_wgs = 10**((2073/self.F1.T) - 2.029)
        # kinetic factor
        k_rwgs = 1.22e10*np.exp(-94765/(self.R*self.F1.T)) #mol/kg_cat*s*bar
        r_num = k_rwgs*self.F1.Xi[2]*self.F1.P*(1 - K_wgs*self.F1.Xi[4]*self.F1.Xi[3]/(self.F1.Xi[0]*self.F1.Xi[2]))
        r_den = (1 + self._K_a*(self.F1.Xi[4]/self.F1.Xi[0]) + self._K_b*(self.F1.Xi[0]*self.F1.P)**0.5 + self._K_c*self.F1.Xi[4]*self.F1.P) 
        return r_num/r_den 

    def MBcheck(self)->float:
        self.m0:float = self.F0.n*self.F0.mw
        self.m1:float = self.F1.n*self.F1.mw
        MB:float = self.m1 - self.m0
        print(f"{(self.F1.n*self.F1.mw/(self.F0.n*self.F0.mw) - 1)*100}% difference")
        return MB
    
    def ExBs(self)->None:
        self.Exth:float = self.Q*(1 - (298.15/self.F1.T))
        self.Exloss:float = self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth
        self.Exeff:float = 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.maximum(0, self.Exth))

    def f_sol(self, x): 
        self.F1.n = x[0]
        self.F1.Xi = x[1:8]
        return self.Eqs()
        
    def run_sol(self, guess_x0=np.array([0.709,0.048,0.074,0.031,0.023,0.051, 0.064])):
        x0 = np.concatenate(([self.F0.n*0.9], guess_x0)) 
        sol = spo.least_squares(self.f_sol, x0)
        self.f_sol(sol.x)
        return sol

#WGS kinetics
class MeOH_WGS(EquilBase):
    """
    Methanol synthesis adiabatic reactor where the methanol yield is given and the WGS reaction is in equilibrium 
    Inlet and outlet tempeartures can be different
    Two reactions are considered:
        1) CO + 2H2 <-> CH3OH  (yield)
        2) CO2 + 3H2 <-> CH3OH + H2O (yield)
        3) CO + H2O <-> CO2 + H2 (equilibrium) WGS
    
    Parameters:
        name: object's name
        Medium: media of the inlet stream. Defaults to media.Fuel_MeOH()
        Yield: methanol yield. Its goes from 0 to 1. Defaults to 0.35
        T: temperature in [K]. Defaults to 473.15 
        P: pressure in [bar]. Defaults to 50.0
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium1. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium2. Defaults to None   
    
    Attributes: 
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        Exth: thermal exergy of the object in [kJ/s] defaulted to 0
        Exloss: exergy loss in the object in [kJ/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Notes:
        Reference: Graaf 1985 "Chemical Equilibria in Methanol Synthesis"
        Water gas shift reaction kinetics are used 
        Methanol yield is based on concentrations of CO and CO2 in the incoming streams.
    
    Examples:
        MeOH_Yield = MeOH_yield.Reactors('MeOH_example', Medium=media.Fuel_MeOH(), Yield=0.33, F0_n=100, 
        F0_Xi=np.array([0.75, 0.05, 0.1, 0.1, 1e-8, 1e-8, 1e-8]), T=473.15, P=50.0)
    
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_MeOH(), 
                 Yield:float=0.35, T:float=473.15, P:float=50.0, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None,  #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None
                 ):
        
        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q)   
        self._N_H = np.array([2,4,0,0,2,0,4])
        self._N_C = np.array([0,1,1,1,0,0,1])
        self._N_O = np.array([0,0,2,1,1,0,1])
        self._N_N = np.array([0,0,0,0,0,2,0])
        self.Yield = Yield
        
    def Eqs(self):

        self.F1.P = self.F0.P
        
        self.Eq:npArray[np.float64] = np.array([
            # MB
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)), # C balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)), # H balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)), # O balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)), # N balance
            self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1], # CH4
            1 - np.sum(self.F1.Xi), # Xi balance       
            # Equil
            self.F1.Xi[3]*self.F1.Xi[4]*self.K_WGS() - (self.F1.Xi[2]*self.F1.Xi[0]), #WGS

            (self.Yield*self.F0.n*(self.F0.Xi[3]+self.F0.Xi[2]) + self.F0.n*self.F0.Xi[6]) - (self.F1.n*self.F1.Xi[6]), #CO and CO2 to MeOH 
            self.Q, #EBal
        ])
        
        return self.Eq

    @property
    def Q(self):
        return self.F0.n*self.F0.h - self.F1.n*self.F1.h 
    
    def K_WGS(self):
        K_WGS = np.exp(- (32.1153*(self.F1.T) - 3.5211e4)/(8.314*self.F1.T))
        return K_WGS
        
    def MBcheck(self)->float:
        MB = self.F1.m - self.F0.m
        print(f"{(self.F1.m/(self.F0.m) - 1)*100}% difference")
        return MB
    
    @property
    def Exth(self):
        return 0 

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth
    
    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n 

    @property
    def conversion(self): 
        H2Conv = ((self.F0.Xi[0])*self.F0.n - (self.F1.Xi[0])*self.F1.n)/((self.F0.Xi[0])*self.F0.n)
        CO2Conv = ((self.F0.Xi[2])*self.F0.n - (self.F1.Xi[2])*self.F1.n)/((self.F0.Xi[2])*self.F0.n)
        CConv = ((self.F0.Xi[2]+self.F0.Xi[3])*self.F0.n - (self.F1.Xi[2]+self.F0.Xi[3])*self.F1.n)/((self.F0.Xi[2]+self.F0.Xi[3])*self.F0.n)
        return {"H2 Conversion": H2Conv, "CO2 Conversion": CO2Conv, "Carbon Conversion": CConv}
        
    def f_sol(self, x): 
        self.F1.n = np.abs(x[0])
        self.F1.Xi = x[1:8]#np.abs(x[1:8])
        self.F1.T = np.abs(x[8])
        return self.Eqs()
        
    def run_sol(self):
        x0 = np.concatenate(([self.F0.n*0.9], np.array([0.69, 0.003, 0.17, 0.09, 0.027, 1e-10, 0.02 ]) , [self.F0.T*1.05])) #self.F0.Xi np.array([0.68, 0, 0.18, 0.046, 0.054,0, 0.04 ]), np.array([0.69, 0.003, 0.17, 0.09, 0.027, 0, 0.02 ]), np.array([0.7, 0.003 , 0.16 , 0.09 , 0.027, 0., 0.02]) media.Fuel_MeOH()._reference_X
        sol = spo.least_squares(self.f_sol, x0)
        self.f_sol(sol.x)
        return sol

class Reformer(ModelBase):
    '''
    Medium = media.media_C3()
    '''
    def __init__(self, name, Medium, F0_n=1., F0_Xi=1., F1_n=1., F1_Xi=None, T=500., P=1., V=1., por=0.75, A = [17400., 23950., 5000.], 
                 Ea = [103191., 0.82*231266., 159241.], beta3=-1, alpha3=2.8, Twall=1000., UA=1e-3):
        
        super().__init__(locals())
        
        self.F0 = interfaces.Flange(name + '.F0', Medium=self.Medium, n=F0_n, Xi=F0_Xi, T=T, P=P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=self.Medium, n=F1_n, Xi=F1_Xi, T=T, P=P)
        
        self.v_i = np.array([
            [1, 0, 1, -1, -1, 0, 0],
            [3, -1, 0, 1, -1, 0, 0],
            [7, 0, 0, 3, -3, 0, -1],
            #[2, 0, -3, 6, 0, 0, -1]
        ], dtype=float)
        self.Eq = np.zeros(self.Medium.nX+2, dtype=float)
        self.r_i = np.zeros(3, dtype=float) #mol/m3cat/s
        
    def Eqs(self): 
        
        self.r_i[0] = self.A[0]*np.exp(-self.Ea[0]/8.314/self.F1.T)*(self.F1.Xi[3]*self.F1.Xi[4]*self.F1.P**2 - (1/self.K_WGS())*self.F1.Xi[2]*self.F1.Xi[0]*self.F1.P**2)
        self.r_i[1] = self.A[1]*np.exp(-self.Ea[1]/8.314/self.F1.T)*(self.F1.Xi[1]*self.F1.Xi[4]*self.F1.P**2 - (1/self.K_MSR())*self.F1.Xi[3]*self.F1.Xi[0]**3*self.F1.P**4)
        self.r_i[2] = self.A[2]*((self.F1.T/1000)**self.beta3)*np.exp(-self.Ea[2]/8.314/self.F1.T)*(self.F1.Xi[6]*self.F1.P)**self.alpha3
        #self.r_i[3] = self.A[3]*np.exp(-self.Ea[3]/8.314/self.F1.T)*(self.F1.Xi[6]*self.F1.P)
        
        self.Q = self.UA*(self.Twall - self.F1.T)
        
        self.Eq[0] = np.sum(self.F1.Xi) - 1. #self.F0.n*self.F0.mw - self.F1.n*self.F1.mw #
        self.Eq[1] = self.F0.n*self.F0.h - self.F1.n*self.F1.h + self.Q
        self.Eq[2:] = self.F0.n*self.F0.Xi - self.F1.n*self.F1.Xi + np.einsum('i, ij->j', self.r_i, self.v_i)*self.V*(1-self.por)
        
        return self.Eq
    
    def Conv(self):
        X = (self.F1.Xi*self.F1.n - self.F0.Xi*self.F0.n)/(self.F0.Xi*self.F0.n)
        return X
    
    def MBcheck(self):
        return self.F0.n*self.F0.mw/(self.F1.n*self.F1.mw) 
    
    def K_PSR(self):
        return np.exp(- (528.68  -0.744*self.F1.T )/(8.314e-3*self.F1.T) ) 
    
    def K_MSR(self):
        return np.exp(- (225_215.699 -252.643*self.F1.T )/(8.314*self.F1.T))
    
    def K_WGS(self):
        return np.exp(- (32.115*(self.F1.T) - 3.521_1E4)/(8.314*self.F1.T))
    
    def Q_WGS(self):
        return self.F1.Xi[2]*self.F1.Xi[0]/self.F1.Xi[3]/self.F1.Xi[4]
    
    def Eq_WGS(self):
        return self.Q_WGS()/self.K_WGS()
    
    def f_sol(self, x):
        self.F1.n = x[0]
        self.F1.T = x[1]
        self.F1.Xi = x[2:]
        
        return self.Eqs()
    
    def run_sol(self, f_method=spo.root):
        x0 = np.concatenate(([self.F0.n, self.F0.T], self.F0.Xi))
        
        sol = f_method(self.f_sol, x0, method='krylov', tol=1e-10)
        #sol = spo.least_squares(self.f_sol, x0, ftol=1e-12, xtol=1e-12, gtol=1e-16)
        self.f_sol(sol.x)
        if not sol.success:
            print(sol.success)
        return sol

class Reformer_1D(SysBase):
    '''
    Medium = media.media_C3()
    '''
    def __init__(self, name, Medium, F0_n=1., F0_Xi=1., T=500., P=1., V=1., N=10, por=0.75, A = [17400., 23950., 5000.], 
                 Ea = [103191., 0.82*231266., 159241.], beta3=-1, alpha3=2.8, Twall=1000., UA=1e-3):
        super().__init__(locals())
        self.Refs = np.array([Reformer(f'{name}[{i}]', Medium=Medium, F0_n = F0_n, F0_Xi=F0_Xi, F1_n=F0_n, F1_Xi=F0_Xi, 
                                       T=T, P=P, V=V/N, por=por, A = A, Ea = Ea, Twall=Twall, UA=UA, beta3=beta3, alpha3=alpha3) for i in range(N)], 
                             dtype=object)
        
    def run_sol(self):
        self.Refs[0]._sol= self.Refs[0].run_sol()
        
        for i in range(self.N-1):
            self.connect_seq(self.Refs[i].F1, self.Refs[i+1].F0)
            self.Refs[i+1]._sol= self.Refs[i+1].run_sol()
            
    def getArrAttr(self, a):
        return np.array([self.Refs[i].rgetattr(a) for i in range(self.N)])
    
    @property
    def Xd(self):
        X = self.getArrAttr('F1.Xi')
        X[:,4] = 0
        X = (1/np.sum(X, axis=1)).reshape(self.N, 1)*X
        return X
