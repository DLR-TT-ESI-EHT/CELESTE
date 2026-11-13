"""
Compressor and turbine unit models with and without performance maps.
"""
import numpy as np
import pandas as pd
import scipy.optimize as spo
from numpy.typing import NDArray as npArray

from .. import ModelBase, interfaces, media
from .corrections import R_UNIVERSAL, Wcorr_from_mf, delta, theta
from .maps import CompressorMap, TurbineMap


class _MappedMixin:
    """
    Behaviour shared by :class:`CompressorMapped` and :class:`TurbineMapped`:
    speed and correction-factor properties, map-or-fixed ``PR``/``eta`` and
    the one-row results table.

    Expects the host class to provide ``F0``, ``Ncorr``, ``beta``, ``Ndesign``,
    ``etaMech``, ``use_map``, ``_PR``, ``_eta`` and a ``_map`` property.
    """

    @property
    def _map(self):
        raise NotImplementedError

    def _q(self):
        """Map state at the current (Ncorr, beta)."""
        return self._map.query(ncorr=self.Ncorr, beta=self.beta, plot=False)

    def _operating_point(self):
        """PR and eta used by ``Eqs``. With ``use_map`` the inlet molar flow is overwritten from the map."""
        if self.use_map:
            q = self._q()
            self.F0.n = q.wcorr*self.delta/np.sqrt(self.theta)*1e3/self.F0.mw
            return q.PR, q.eta
        return self._PR, self._eta       # F0.n stays as given

    # ------------------------------------------------------------------ #
    #  Correction factors and shaft speed                                 #
    # ------------------------------------------------------------------ #
    @property
    def theta(self):
        """float : Temperature correction factor / -"""
        return theta(self.F0.T)

    @property
    def delta(self):
        """float : Pressure correction factor / -"""
        return delta(self.F0.P*1e5)

    @property
    def omega(self):
        """float : Shaft rotational speed / rad/s"""
        return self.Ncorr*np.sqrt(self.theta)*self.Ndesign*(2*np.pi/60)

    @omega.setter
    def omega(self, omega):
        """Set the shaft speed / rad/s by updating ``Ncorr``."""
        self.Ncorr = omega / (np.sqrt(self.theta) * self.Ndesign * (2*np.pi/60))

    @property
    def N(self):
        """float : Shaft rotational speed / rpm"""
        return self.omega * 60 / (2*np.pi)

    def calc_Ndesign(self, omega_design: float, Ncorr_design: float = 1.0) -> float:
        """
        Compute and set ``Ndesign`` from a known design-point shaft speed.

        Parameters
        ----------
        omega_design : float
            Design-point shaft rotational speed / rad/s.
        Ncorr_design : float, default 1.0
            Design-point corrected speed / - (usually 1.0 by definition).

        Returns
        -------
        float
            Design shaft speed / rpm, also stored on ``self.Ndesign``.
        """
        self.Ndesign = omega_design / (Ncorr_design * np.sqrt(self.theta) * (2*np.pi/60))
        return self.Ndesign

    # ------------------------------------------------------------------ #
    #  Map outputs (or fixed values when use_map is False)                #
    # ------------------------------------------------------------------ #
    @property
    def Wcorr(self):
        """float : Corrected mass flow / kg/s"""
        if self.use_map:
            return self._q().wcorr
        return Wcorr_from_mf(self.F0.m * 1e-3, self.F0.T, self.F0.P * 1e5)

    @property
    def m_map(self):
        """float : Mass flow from the map / kg/s"""
        return self.Wcorr * self.delta / np.sqrt(self.theta)

    @property
    def PR(self):
        """float : Pressure ratio / - (compressor: outlet/inlet, turbine: inlet/outlet)"""
        if self.use_map:
            return self._q().PR
        return self._PR

    @PR.setter
    def PR(self, value):
        self._PR = value

    @property
    def eta(self):
        """float : Isentropic efficiency / -"""
        if self.use_map:
            return self._q().eta
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = value

    # ------------------------------------------------------------------ #
    #  Reporting / solving                                                #
    # ------------------------------------------------------------------ #
    def _map_outputs(self) -> dict:
        return {
            "Wcorr /kg/s": self.Wcorr,
            "m_map /kg/s": self.m_map,
            "PR": self.PR,
            "eta": self.eta,
        }

    def results_df(self) -> pd.DataFrame:
        """
        Return all relevant calculated quantities as a one-row DataFrame.

        Columns: operating point (``Ncorr``, ``beta``, ``theta``, ``delta``),
        map outputs, speeds, performance (``tau``, ``Pshaft``) and flows.
        """
        data = {
            "name": self.name,
            "Ndesign": self.Ndesign,

            # Operating point
            "Ncorr": self.Ncorr,
            "beta": self.beta,
            "theta": self.theta,
            "delta": self.delta,

            # Map outputs
            **self._map_outputs(),
            "eta mech": self.etaMech,

            # Speeds
            "omega /rad/s": self.omega,

            # Performance
            "tau /Nm": self.tau,
            "Pshaft /W": self.Pshaft,

            # Flow
            "m /kg/s": self.F0.m*1000,
            "n /mol/s": self.F0.n,
        }
        return pd.DataFrame([data])

    def run_sol(self):
        """Solve the component by evaluating its equations (alias for :meth:`Eqs`)."""
        return self.Eqs()

   
class Compressor(ModelBase):
    """
    base.ModelBase object with one flange F0 in and one flange F1 out. 
    This is the base Compressor object from which other related objects could extent. No methods are included.
    
    Parameters:
        name: objects name
        Medium: media of the streams
        eff: work efficiency of the Compressor object. Must be 0 < eff <= 1. Defaults to 1
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium. Defaults to None   
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: outlet temperature in [K]. Defaults to 398.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: outlet pressure in [bar]. Defaults to 3.0 

    Attributes: 
        F0: flange F0
        F1: flange F1
    """
    def __init__(self, name:str, Medium:media.Medium_Base, eff:float=None,
                F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0,
                F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, F1_T:float=398.15, F1_P:float=3.0, 
                ):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P)

class CompIsot(Compressor):
    """
    Isothermal compressor extending from the Compressor class. 
    One stream is entering and one stream is coming out. 
    The outlet n, Xi, h, cp, g, s are determined.
    The work of the CompIsot object is calculated isothermally.
    IMPORTANT: the inlet temperature and pressure, and the outlet pressure must be given manually.
    
    Parameters:
        name: objects name
        Medium: media of the streams
        eff: work efficiency of the CompIsot object. Must be 0 < eff <= 1. Defaults to 1
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium. Defaults to None   
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: outlet temperature in [K]. Defaults to 398.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: outlet pressure in [bar]. Defaults to 3.0 
    
    Attributes: 
        R: ideal gas constant in [kJ/mol K]
        W: work in [kJ/s]

    Methods: 
        Eqs: the outlet mol flow, composition, temperature, work and heat are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        run_sol: empty method. Returns: the system of equations in the "Eqs" method

    Notes:  
        The size of F0_Xi and F1_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.

    Examples:
        1) 
        Compressor = Components.CompIsot('Compressor_example', Medium=media.Fuel_CH4(), F0_n=100, 
        F0_Xi=np.array([0.1,1e-8,1e-8,1e-8,0.9,1e-8]), F0_T=473.15, F0_P=3.0, F1_P=6.0)

    """
    def __init__(self, name:str, Medium:media.Medium_Base, eff=1.,
                 F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0,
                 F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, F1_T:float=298.15, F1_P:float=3.0
                 ):
        super().__init__(name, Medium=Medium, eff=eff, F0_n=F0_n, F0_Xi=F0_Xi, F0_T=F0_T, F0_P=F0_P, 
                         F1_n=F1_n, F1_Xi=F1_Xi, F1_T=F1_T, F1_P=F1_P)
        self.R = 8.314462618e-3
        
    def Eqs(self)->None:
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        self.F1.T = self.F0.T
    
    @property
    def W(self):
        return self.F0.n*self.F0.T*self.R*np.log(self.F1.P/self.F0.P)/self.eff

    @property
    def Q(self):
        return -self.W

    @property
    def Exth(self):
        return self.Q*(1 - 298.15/self.F0.T)
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.W + self.Exth

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n - self.Q/self.F0.T
    
    def MBcheck(self)->float:
        # Mass balance
        print(f"{abs((self.F0.m/self.F1.m) - 1)*100}% difference ")    
        return self.F0.m - self.F1.m
        
    def run_sol(self):
        return self.Eqs()
 
class CompIsent(Compressor):
    """
    Isentropic compressor extending from the Compressor Class. 
    One stream is entering and one stream is coming out. 
    The outlet n, Xi, h, cp, g, s, T are determined.
    The real work of the CompIsent object is calculated for a given efficiency.  
    IMPORTANT: the inlet temperature and pressure, and the outlet pressure must be given manually.
    
    Parameters:
        name: objects name
        Medium: media of the streams
        eff: work efficiency of the CompIsent object. Must be 0 < eff <= 1. Defaults to 1
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium. Defaults to None   
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: outlet temperature in [K]. Defaults to 398.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: outlet pressure in [bar]. Defaults to 3.0 
    
    Attributes: 
        R: ideal gas constant in [kJ/mol K]
        cpa: average cp between the inlet and outlet streams in [kJ/mol K]
        Eq: equation to be solved
        W: real work in [kJ/s]
        Exloss: exergy loss in the object [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol flow and composition are determined. Returns: "Eq" attribute
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        f_sol: contains the variable to be optimized by the solver. Return: the "Eqs" method
        run_sol: contains the optimizer function and guesses values. Returns: the solution of the optimization

    Notes:  
        The size of F0_Xi and F1_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.

    Examples:
        1) 
        Compressor = Components.CompIsent('Compressor_example', Medium=media.Fuel_CH4(), eff=0.7, F0_n=100, 
        F0_Xi=np.array([0.1,1e-8,1e-8,1e-8,0.9,1e-8]), F0_T=473.15, F0_P=3.0, F1_P=6.0)

    """
    def __init__(self, name:str, Medium:media.Medium_Base, eff:float=1,
                 F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0,
                 F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, F1_T:float=398.15, F1_P:float=3.0
                 ):
        super().__init__(name, Medium=Medium, eff=eff, F0_n=F0_n, F0_Xi=F0_Xi, F0_T=F0_T, F0_P=F0_P, 
                         F1_n=F1_n, F1_Xi=F1_Xi, F1_T=F1_T, F1_P=F1_P)
        self.R = 8.314462618e-3
        
    def Eqs(self):
        cpa = np.average([self.F0.cp, self.F1.cp])
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        if self.F1.P > self.F0.P:
            eff_ratio = 1/self.eff
        else:
            eff_ratio = self.eff
        self.Eq = self.F1.T - (self.F0.T + (self.F0.T*(self.F1.P/self.F0.P)**(self.R/cpa) - self.F0.T)*eff_ratio )        
        return self.Eq

    @property
    def W(self):
        return self.F0.n*(self.F1.h - self.F0.h)
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.W

    @property
    def Exeff(self):
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + self.W)

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n
    
    def MBcheck(self)->float:
        # Mass balance
        print(f"{abs((self.F0.m / self.F1.m ) - 1)*100}% difference ")    
        return self.F0.m - self.F1.m

    def f_sol(self, x):
        self.F1.T = x[0]
        return self.Eqs()
        
    def run_sol(self):
        x0 = self.F0.T*(self.F1.P/self.F0.P)**(self.R/1.3)
        sol = spo.root(self.f_sol, x0)
        self.f_sol(sol.x)
        if not sol.success:
            print(f'{self.name} Failed')
        return sol


class CompressorMapped(_MappedMixin, Compressor):
    """
    Compressor model with optional performance-map lookup.

    With ``use_map=False`` the pressure ratio and isentropic efficiency are
    fixed values and the inlet molar flow is taken as given. With
    ``use_map=True`` the pressure ratio, efficiency and corrected mass flow
    are read from a compressor map at the current corrected speed and beta
    line, and the inlet molar flow is overwritten accordingly.

    Parameters
    ----------
    name : str
        Name of the component.
    comp_map : CompressorMap
        Map providing ``query(ncorr, beta, plot)``.
    Medium : media.Medium_Base
        Medium used for the inlet, outlet and isentropic flanges.
    F0_n, F0_Xi, F0_T, F0_P
        Inlet molar flow / mol/s, composition, temperature / K, pressure / bar.
    F1_n, F1_Xi, F1_T, F1_P
        Initial guess for the outlet state.
    PR : float, default 1.0
        Fixed pressure ratio / -. Used only when ``use_map`` is False.
    eta : float, default 0.9
        Fixed isentropic efficiency / -. Used only when ``use_map`` is False.
    Ncorr : float, default 1.0
        Corrected speed relative to the design speed / -.
    beta : float, default 0.5
        Beta-line parameter locating the operating point on the map / -.
    Ndesign : float, default 0.0
        Design shaft speed / rpm. Must be positive for ``omega``, ``N`` and
        ``tau`` to be meaningful (see :meth:`calc_Ndesign`).
    etaMech : float, default 1.0
        Mechanical efficiency of the shaft / -.
    use_map : bool, default False
        If True, take ``PR``, ``eta`` and the flow from the compressor map.

    Attributes
    ----------
    h_real : float
        Real outlet specific enthalpy, set in ``Eqs``.
    Pshaft : float
        Shaft power / W, set in ``Eqs``.
    tau : float
        Shaft torque / Nm, set in ``Eqs``. NaN if ``omega`` is zero.
    """
    def __init__(self, name: str, comp_map: CompressorMap, Medium: media.Medium_Base,
                 F0_n: float = 1.0, F0_Xi=None, F0_T: float = 298.15, F0_P: float = 1.0,
                 F1_n: float = 1.0, F1_Xi=None, F1_T: float = 398.15, F1_P: float = 3.0,
                 PR: float = 1., eta: float = 0.9, Ncorr=1., beta=0.5, Ndesign=0.,
                 etaMech: float = 1., use_map: bool = False):
        super().__init__(name, Medium=Medium, eff=eta, F0_n=F0_n, F0_Xi=F0_Xi, F0_T=F0_T, F0_P=F0_P,
                         F1_n=F1_n, F1_Xi=F1_Xi, F1_T=F1_T, F1_P=F1_P)

        # Stored with a leading underscore so ModelBase.to_dict()/to_json() skip the map object
        self._comp_map = comp_map
        self.__F = interfaces.Flange(name + '.F1_iso', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P)

        self.Ncorr = Ncorr
        self.beta = beta
        self.Ndesign = Ndesign
        self.etaMech = etaMech
        self._PR = PR
        self._eta = eta
        self.use_map = use_map

    @property
    def comp_map(self) -> CompressorMap:
        """CompressorMap : The compressor map used for lookups."""
        return self._comp_map

    @comp_map.setter
    def comp_map(self, value: CompressorMap):
        self._comp_map = value

    @property
    def _map(self):
        return self._comp_map

    def Eqs(self):
        """
        Evaluate the compressor equations.

        The outlet pressure follows from the pressure ratio, the isentropic
        outlet state is computed on an internal flange, and the real outlet
        enthalpy is obtained from the isentropic efficiency:

        .. math:: h_{real} = h_0 + (h_{1,s} - h_0) / \\eta

        The shaft power is

        .. math:: P_{shaft} = \\dot{n} (h_1 - h_0) / \\eta_{mech}
        """
        PR, eta = self._operating_point()
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        self.F1.P = self.F0.P*PR
        self.__F.T_sPX(self.F0.s, self.F1.P, self.F1.Xi)
        self.h_real = (self.__F.h - self.F0.h)/eta + self.F0.h
        self.F1.T_hx(self.h_real, self.F1.Xi)
        self.Pshaft = self.F0.n*(self.F1.h - self.F0.h)*1e3/self.etaMech   # W
        self.tau = self.Pshaft/self.omega if self.omega > 0 else np.nan

    @property
    def surge(self) -> bool:
        """bool : True if the current operating point lies in the surge region"""
        return self._q().in_surge

    def _map_outputs(self) -> dict:
        out = super()._map_outputs()
        out["surge"] = self.surge
        return out


class Turbine(ModelBase):
    """
    Base turbine model with one inlet flange (F0) and one outlet flange (F1).

    Parameters
    ----------
    name : str
        Name of the component.
    Medium : media.Medium_Base
        Medium (fluid) model of the streams.
    eff : float, optional
        Work efficiency, 0 < eff <= 1. Currently not used to build the flanges.
    F0_n, F0_Xi, F0_T, F0_P
        Inlet molar flow / mol/s, composition, temperature / K, pressure / bar.
    F1_n, F1_Xi, F1_T, F1_P
        Outlet molar flow / mol/s, composition, temperature / K, pressure / bar.
    """
    def __init__(self, name: str, Medium: media.Medium_Base, eff: float = None,
                 F0_n: float = 1.0, F0_Xi: npArray[np.float64] | None = None, F0_T: float = 298.15, F0_P: float = 1.0,
                 F1_n: float = 1.0, F1_Xi: npArray[np.float64] | None = None, F1_T: float = 398.15, F1_P: float = 3.0):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P)


class TurbineMapped(_MappedMixin, Turbine):
    """
    Turbine model with optional performance-map lookup.

    With ``use_map=True`` the pressure ratio, isentropic efficiency and
    corrected mass flow are read from a turbine map at the current corrected
    speed and beta line, and the inlet molar flow is overwritten accordingly.
    With ``use_map=False`` the pressure ratio and efficiency are fixed values
    and the inlet molar flow is taken as given.

    Parameters
    ----------
    name : str
        Name of the component.
    turb_map : TurbineMap
        Map providing ``query(ncorr, beta, plot)``.
    Medium : media.Medium_Base
        Medium used for the inlet, outlet and isentropic flanges.
    F0_n, F0_Xi, F0_T, F0_P
        Inlet molar flow / mol/s, composition, temperature / K, pressure / bar.
    F1_n, F1_Xi, F1_T, F1_P
        Initial guess for the outlet state.
    PR : float, default 1.0
        Fixed expansion pressure ratio (inlet/outlet) / -. Only used when ``use_map`` is False.
    eta : float, default 1.0
        Fixed isentropic efficiency / -. Only used when ``use_map`` is False.
    Ncorr : float, default 1.0
        Corrected speed relative to the design speed / -.
    beta : float, default 0.5
        Beta-line parameter locating the operating point on the map / -.
    Ndesign : float, default 0.0
        Design shaft speed / rpm (see :meth:`calc_Ndesign`).
    etaMech : float, default 1.0
        Mechanical efficiency of the shaft / -.
    Mach_out : float, default 0.2
        Design outlet Mach number, converts between static and total outlet
        pressure / -. Superseded by the value computed from ``A_exit`` once
        ``A_exit`` is set.
    A_exit : float, optional
        Geometric exit area / m². Set it directly or with :meth:`size_exit_area`.
    blockage : float, default 0.03
        Exit-area blockage fraction / -.
    use_map : bool, default True
        If True, take ``PR``, ``eta`` and the flow from the turbine map.

    Attributes
    ----------
    h_real : float
        Real outlet specific enthalpy, set in ``Eqs``.
    Pshaft : float
        Shaft power / W, set in ``Eqs``. Negative means the turbine delivers power.
    tau : float
        Shaft torque / Nm, set in ``Eqs``. NaN if ``omega`` is zero.
    """
    def __init__(self, name: str, turb_map: TurbineMap, Medium: media.Medium_Base,
                 F0_n: float = 1.0, F0_Xi=None, F0_T: float = 298.15, F0_P: float = 1.0,
                 F1_n: float = 1.0, F1_Xi=None, F1_T: float = 398.15, F1_P: float = 3.0,
                 PR: float = 1., eta: float = 1., Ncorr=1., beta=0.5, Ndesign=0., etaMech: float = 1.,
                 Mach_out: float = 0.2, A_exit: float = None, blockage: float = 0.03,
                 use_map: bool = True):
        super().__init__(name, Medium=Medium, F0_n=F0_n, F0_Xi=F0_Xi, F0_T=F0_T, F0_P=F0_P,
                         F1_n=F1_n, F1_Xi=F1_Xi, F1_T=F1_T, F1_P=F1_P)

        # Stored with a leading underscore so ModelBase.to_dict()/to_json() skip the map object
        self._turb_map = turb_map
        self.__F = interfaces.Flange(name + '.F1_iso', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P)

        self.Ncorr = Ncorr
        self.beta = beta
        self._PR = PR
        self._eta = eta
        self.Ndesign = Ndesign
        self.etaMech = etaMech
        self._Mach_out = Mach_out      # design Mach (used for sizing only if A_exit is None)
        self.A_exit = A_exit           # m², geometric exit area
        self.blockage = blockage
        self.use_map = use_map

    @property
    def turb_map(self) -> TurbineMap:
        """TurbineMap : The turbine map used for lookups."""
        return self._turb_map

    @turb_map.setter
    def turb_map(self, value: TurbineMap):
        self._turb_map = value

    @property
    def _map(self):
        return self._turb_map

    # ------------------------------------------------------------------ #
    #  Exit Mach number / total outlet pressure                           #
    # ------------------------------------------------------------------ #
    @property
    def Mach_out(self):
        """Fixed design value until A_exit exists, then computed from the flow."""
        return self._Mach_out if self.A_exit is None else self.mach_from_area()

    @Mach_out.setter
    def Mach_out(self, value):
        self._Mach_out = value

    def _exit_static(self):
        R = R_UNIVERSAL / (self.F1.mw * 1e-3)       # J/kg/K
        return R, self.F1.T, self.F1.P * 1e5, self.F1.m * 1e-3   # R, T /K, p /Pa, mdot /kg/s

    def mach_from_area(self):
        R, T, p, mdot = self._exit_static()
        A_eff = self.A_exit * (1 - self.blockage)
        return mdot / (A_eff * p) * np.sqrt(R * T / self.gamma)

    def size_exit_area(self):
        """Design point: size A_exit from the chosen Mach number, then keep it fixed."""
        R, T, p, mdot = self._exit_static()
        self.A_exit = mdot * np.sqrt(R * T / self.gamma) / (p * self._Mach_out * (1 - self.blockage))
        return self.A_exit

    @property
    def gamma(self):
        """float : Isentropic exponent / -"""
        cp_molar = self.F1.cp*1e3
        cv_molar = cp_molar - R_UNIVERSAL
        return cp_molar / cv_molar

    @property
    def p0(self):
        """float : Total outlet pressure / bar"""
        return self.F1.P * (1 + (self.gamma-1)/2*self.Mach_out**2)**(self.gamma/(self.gamma-1))

    @p0.setter
    def p0(self, p0):
        """Set the total outlet pressure / bar by updating the static outlet pressure ``F1.P``."""
        self.F1.P = p0 / ((1 + (self.gamma-1)/2*self.Mach_out**2)**(self.gamma/(self.gamma-1)))

    # ------------------------------------------------------------------ #
    #  Equations and design solver                                        #
    # ------------------------------------------------------------------ #
    def Eqs(self) -> None:
        """
        Evaluate the turbine equations.

        The outlet pressure follows from the pressure ratio, the isentropic
        outlet state (constant inlet entropy, pressure ``F0.P / PR``) is
        computed on an internal flange, and the real outlet enthalpy is

        .. math:: h_{real} = h_0 + \\eta (h_{1,s} - h_0)

        The shaft power is

        .. math:: P_{shaft} = \\dot{n} (h_1 - h_0) \\eta_{mech}

        which is negative when the turbine delivers power.
        """
        PR, eta = self._operating_point()
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        self.F1.P = self.F0.P/PR
        self.__F.T_sPX(self.F0.s, self.F1.P, self.F1.Xi)       # isentropic outlet
        self.h_real = (self.__F.h - self.F0.h)*eta + self.F0.h
        self.F1.T_hx(self.h_real, self.F1.Xi)
        self.Pshaft = self.F0.n*(self.F1.h - self.F0.h)*1e3*self.etaMech   # W, negative = power delivered
        self.tau = self.Pshaft/self.omega if self.omega > 0 else np.nan

    def solve_design_PR(self, Pshaft_target: float,
                        PR_bounds=(1.001, 50.), xtol=1e-12, verbose: int = 0):
        """
        Find the pressure ratio that delivers a target shaft power.

        The map is disabled (``use_map`` is set to False) and the solution
        does not depend on any previous state. On success the turbine is left
        in the solved state.

        Parameters
        ----------
        Pshaft_target : float
            Target shaft power delivered / W, given as a positive value.
        PR_bounds : tuple of float, default (1.001, 50.)
            Lower and upper bracket for the pressure ratio search.
        xtol : float, default 1e-12
            Absolute tolerance on the pressure ratio for :func:`scipy.optimize.brentq`.
        verbose : int, default 0
            If >= 1, print a summary of the solution.

        Returns
        -------
        scipy.optimize.OptimizeResult
            ``x`` (solved pressure ratio as a one-element array), ``fun``
            (relative residual), ``success`` and ``nfev``.

        Raises
        ------
        ValueError
            If the residuals at the bounds are not finite or do not bracket a root.
        """
        self.use_map = False

        def res(PR):
            self.PR = PR
            self.run_sol()
            return (-self.Pshaft - Pshaft_target) / Pshaft_target

        lo, hi = PR_bounds
        r_lo, r_hi = res(lo), res(hi)
        if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo * r_hi > 0:
            raise ValueError(f'{self.name}: no PR in {PR_bounds} delivers Pshaft={Pshaft_target:.4g} '
                             f'(residuals {r_lo:.3g}, {r_hi:.3g})')

        PR, info = spo.brentq(res, lo, hi, xtol=xtol, rtol=1e-12, full_output=True)
        r = res(PR)                      # leave the turbine in the solved state

        if verbose >= 1:
            print(f"[{self.name}] Design PR solved: PR={PR:.6f}, residual={r:.2e}, "
                  f"Pshaft={self.Pshaft:.1f} W (target {Pshaft_target:.1f} W)")

        return spo.OptimizeResult(x=np.array([PR]), fun=np.array([r]),
                                  success=bool(info.converged), nfev=info.function_calls)
