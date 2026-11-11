import inspect

from celeste import CSL


class IntrospectionMixin:
    """Hierarchy traversal and component lookup (sources, sinks, heaters, flanges)."""

    def r_get(self, prefix=''):
        """
        Recursively retrieves the names of all attributes within the units in system.

        This method traverses the attributes of the current system, and for each attribute that is an instance of ModelBase,
        it adds its name to the output list. If the attribute is also an instance of SysBase (i.e., a subsystem),
        the method is called recursively on that subsystem, and the names are prefixed accordingly.

        Args:
            prefix (str, optional): String to prepend to each attribute name, used for hierarchical naming in nested systems.
                                    Defaults to ''.

        Returns:
            list of str: A list of dot-separated attribute names representing all ModelBase instances in the system hierarchy.
        """
        from .._base import ModelBase
        out = []
        for i in inspect.getmembers(self):
            if isinstance(i[1], ModelBase):
                if getattr(i[1], "_is_system", False):
                    out += i[1].r_get(prefix + i[0] + '.')
                else:
                    out += [prefix + i[0]]
        return out

    def get_AllFlanges(self):
        """
        Retrieves all flange objects (flow interfaces) within the system hierarchy.

        Recursively searches through all ModelBase components and subsystems, collecting
        every flange (flow connection point) defined in the system.

        Returns:
            list: A list of Flange objects from all components in the system.
        """
        flanges = []
        for i in self.r_get():
            flanges += self.rgetattr(i).get_Flanges()
        return flanges

    def ViewAllFlange(self):
        """
        Provides a summarized view of all flanges in the system.

        Returns:
            pandas.DataFrame or str: A structured view showing key properties of all flanges,
            typically including flow rates, pressures, temperatures, and compositions.
        """
        return self.ViewFlange(self.get_AllFlanges())

    def get_success(self):
        """
        Compiles the convergence status of all numerical solvers in the system.

        Checks components that have an 'f_sol' method (indicating they perform numerical solving)
        and returns their solver success status.

        Returns:
            dict: A dictionary mapping component paths to boolean success flags.
                True indicates successful convergence, False indicates failure.
        """
        success = {}
        for i in self.r_get():
            if 'f_sol' in dir(self.rgetattr(i)):
                success[i] = self.rgetattr(i)._sol.success
        return success

    def get_Sources(self):
        """
        Identifies all source components in the system hierarchy.

        Sources are defined as components that introduce material/energy flows into the system
        (instances of CSL.flow.SourceM).

        Returns:
            dict: A dictionary mapping source component paths to their object references.
        """
        sources = {}
        for i in self.r_get():
            if isinstance(self.rgetattr(i), CSL.flow.SourceM):
                sources[i] = self.rgetattr(i)
        return sources

    def get_Sinks(self):
        """
        Identifies all sink components in the system hierarchy.

        Sinks are defined as components that remove material/energy flows from the system
        (instances of CSL.flow.Sink).

        Returns:
            dict: A dictionary mapping sink component paths to their object references.
        """
        sinks = {}
        for i in self.r_get():
            if isinstance(self.rgetattr(i), CSL.flow.Sink):
                sinks[i] = self.rgetattr(i)
        return sinks

    def get_Heaters(self):
        heaters = {}
        for i in self.r_get():
            if isinstance(self.rgetattr(i), CSL.heat.Heater):
                heaters[f'{self.name}.{i}'] = self.rgetattr(i)
        return heaters

    def get_Heat(self):
        heat = {}
        for i in self.r_get():
            if 'Q' in dir(self.rgetattr(i)) and not isinstance(
                self.rgetattr(i), (CSL.heat.Heater, CSL.heat.HexBase, CSL.components.DAC)
            ):
                heat[f'{self.name}.{i}'] = self.rgetattr(i)
        return heat

    def get_Exloss(self):
        exloss = {}
        for i in self.r_get():
            if not isinstance(self.rgetattr(i), (CSL.flow.SourceM, CSL.flow.Sink)):
                exloss[f'{self.name}.{i}'] = self.rgetattr(i)
        return exloss

    def SBcheck(self):
        sg = {}
        for i in self.r_get():
            if not isinstance(self.rgetattr(i), (CSL.flow.SourceM, CSL.flow.Sink)):
                sgen = self.rgetattr(i).Sgen #np.where(np.abs(self.rgetattr(i).Sgen) > 1e-10, self.rgetattr(i).Sgen, 0)
                sg[i] = {
                    'Sgen': sgen,
                    'Sgen*T0': sgen*298.15
                    }
        return pd.DataFrame(sg).transpose()
    
    def exdest_df(self):
        comp = 0
        SOC = 0
        heaters = 0
        coolers = 0
        reactor = 0
        valves = 0
        gpu = 0
        sp_mx = 0
        ASU = 0
        DAC = 0
        tot = self.tot_Exloss()
        for i,v in self.get_Exloss().items():
            if isinstance(self.rgetattr(i), (CSL.components.Pump, CSL.components.Compressor)):
                comp += v.Exloss
            if isinstance(self.rgetattr(i), CSL.electrochem.SOC):
                SOC += v.Exloss
            if isinstance(self.rgetattr(i), CSL.components.ASU):
                ASU += v.Exloss
            if isinstance(self.rgetattr(i), CSL.components.DAC):
                DAC += v.Exloss
            if isinstance(self.rgetattr(i), CSL.heat.Heater):
                heaters += np.where(v.Q > 0, v.Exloss, 0) 
                coolers += np.where(v.Q < 0, v.Exloss, 0) 
            if isinstance(self.rgetattr(i), CSL.reactors.MR): # have to add the other isothermal reactors
                reactor += v.Exloss 
            if isinstance(self.rgetattr(i), (CSL.components.GPU_2S,
                                            CSL.components.GPU_2SV2, 
                                            CSL.components.GPU_3S)):
                gpu += v.Exloss 
            if isinstance(self.rgetattr(i), (CSL.flow.Mixer, CSL.flow.Splitter)):
                sp_mx += v.Exloss
            if 'valve' in i or 'Valve' in i:
                valves += v.Exloss
        final = {'Compressors':comp,
                'SOC': SOC,
                'ASU': ASU,
                'DAC': DAC,
                'Heaters': heaters,
                'Coolers': coolers,
                'GPU': gpu,
                'Reactor':reactor,
                'Split_Mix':sp_mx,
                'Valves':valves,
                'Total': tot,
                }
        ex = pd.DataFrame({'Ex. destruction / kW':final})
        ex.insert(1, '%', ex.values*100/tot)
        return ex
