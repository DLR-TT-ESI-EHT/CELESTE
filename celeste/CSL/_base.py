import inspect

import numpy as np
import pandas as pd
import scipy.optimize as spo

from .interfaces import Flange
from .. import Base

from ._mixin.introspection import IntrospectionMixin
from ._mixin.graph import GraphMixin
from ._mixin.analysis import ReportingMixin
from ._mixin.recycle import RecycleMixin

from typing import Any
from numpy._typing._array_like import NDArray

class ModelBase(Base):
    '''
    Base for creating models such as pins and vessels with flexible attribute definition
    '''

    def __init__(self, args):
        super().__init__(args)
        self.Eq = np.array([])

    def sol_seq(self):
        return None

    def Eqs(self) -> list[float] | NDArray[np.float64]:
        return []
    
    @property
    def Exloss(self) -> float:
        return 0.

    @staticmethod
    def ViewFlange(flanges):
        """
        View details of flanges in a tabular format.

        Args:
            flanges (Union[ModelBase, List[Flange]]): A ModelBase object or a list of Flange objects.

        Returns:
            pd.DataFrame: DataFrame containing flange details.
        
        Raises:
            ValueError: If an invalid object is passed.
        """
        if isinstance(flanges, ModelBase):
            flangeList = flanges.get_Flanges()
        elif isinstance(flanges, list) and all(hasattr(f, "get_nXTP") for f in flanges):
            flangeList = flanges
        else:
            raise ValueError("Invalid input: Expected a ModelBase object or a list of Flange objects.")

        return (pd.DataFrame([f.get_nXTP() for f in flangeList])
                .transpose()
                .fillna(0)
                #.clip(lower=1e-8)
        )

    def get_Flanges(self) -> list[Flange]:
        """
        Get all Flange objects associated with this model.

        Returns:
            List[Flange]: A list of Flange objects.
        """
        return [attr for _, attr in inspect.getmembers(self) if isinstance(attr, Flange)]



class SysBase(GraphMixin, ReportingMixin, IntrospectionMixin, RecycleMixin, ModelBase):
    _is_system=True
    def __init__(self, args): # name,
        super().__init__(args) #

        self.Eq = np.array([])
        self.Objs = np.array([], dtype=ModelBase)
        self._core = ['name', 'self','Medium']
        self.connections = []

    def connect(self, pin1, pin2) -> None:
        """

        Args:
            pin1 (_type_): _description_
            pin2 (_type_): _description_

        Raises:
            Exception: _description_
            Exception: _description_
        """
        Eq = np.array([])
        # check if correct pins
        if pin1!=pin2: 
            raise ValueError(f"Incorrect connection between {pin1.name} and {pin2.name}")
        
        for attr in pin1.__dict__.keys(): #for each flange of the other components 
            if any(k in attr for k in self._core):
                continue 

            if attr in ['Q','I']: # Flow variables sum to zero at interface
                Eq = np.append(Eq, [pin1.__dict__[attr] + pin2.__dict__[attr]])
            elif attr in ['n', 'T', 'P']: # add equations describing the equality on the flange
                Eq = np.append(Eq, [ pin1.__dict__[attr] - pin2.__dict__[attr] ])
            
        self.Eq = np.append(self.Eq, Eq)

    def connect_seq(self, pin1:Flange, pin2:Flange, direction:str='F', store:bool=False) -> None:
        '''
        forces assignment of one pins conditions to the other

        direction, str = F for forward, B for backward
        '''

        if pin1!=pin2:  
            raise ValueError(f"Incorrect connection between {pin1.name} and {pin2.name}")
        
        if store:
            self.connections.append(['.'.join(pin1.name.split('.')[:-1]), pin1.name.split('.')[-1], pin2.name.split('.')[-1], '.'.join(pin2.name.split('.')[:-1])])
        
        for var in pin1._pin_vars:
            if var in pin1._flow_vars:
                if direction.lower() == 'b':
                    setattr(pin1, var, -getattr(pin2, var))
                else:
                    setattr(pin2, var, -getattr(pin1, var))
            else:
                if direction.lower() == 'b':
                    setattr(pin1, var, getattr(pin2, var))
                else:
                    setattr(pin2, var, getattr(pin1, var))

    def getObjs(self) -> list[ModelBase]:
        """
        Get all objects added to the system.

        Returns:
            List[object]: List of objects added to the system.
        """
        return [v for v in self.__dict__.values() if isinstance(v, ModelBase)]    

    def makeSys(self) -> None:
        """
        Aggregate equations from all objects in the system.
        """
        self.Eq = np.concatenate([obj.Eqs() for obj in self.getObjs()])

    @staticmethod
    def GradientDescent(fun, x, r=1e-2, m=0.03, eps=1e-6, Niter=5, print_steps=False):
        """
        Perform gradient descent optimization.

        Args:
            fun (callable): Objective function.
            x (np.ndarray): Initial guess.
            r (float): Learning rate.
            m (float): Momentum factor.
            eps (float): Perturbation size for gradient approximation.
            Niter (int): Number of iterations.
            print_steps (bool): Whether to print intermediate steps.

        Returns:
            np.ndarray: Optimized solution.
        """
        change = 0
        for i in range(Niter):
            print(i)
            f_val = fun(x)
            grad_approx = spo.approx_fprime(x, fun, eps)
            grad_step = np.linalg.inv(grad_approx) @ f_val
            
            new_change = r * grad_step + m * change
            x -= new_change
            change = new_change
            if print_steps:
                print(f"Iteration {i}: f={f_val}, g={new_change}")

            print('#'*10)

        return x
    
    def MBcheck(self):
        sources = self.get_Sources().values()
        sinks = self.get_Sinks().values()
        
        m_in = 0
        for i in sources:
            m_in += i.F1.m

        m_out = 0
        for i in sinks:
            m_out += i.F0.m

        return m_out - m_in

    def EBcheck(self):
        sources = self.get_Sources().values()
        sinks = self.get_Sinks().values()
        
        self.q_in = 0
        for i in sources:
            self.q_in += i.F1.n*i.F1.h

        self.q_out = 0
        for i in sinks:
            self.q_out += i.F0.n*i.F0.h

        return (self.tot_heaters_in() + self.tot_heaters_out() + self.tot_heat_in() + self.tot_heat_out() + 
                self.tot_Qloss() + self.tot_work_in() + self.tot_work_out() + self.q_in - self.q_out
              )
    
