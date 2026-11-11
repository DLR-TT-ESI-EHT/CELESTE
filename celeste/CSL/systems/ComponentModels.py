from celeste.CSL import interfaces, media, heat, flow, components, SysBase

class SimpleHeaterSys(SysBase):
    def __init__(self, name, flow_in, Xi_in, T_in, P_in, T_out):
        self.__init__(locals())
        ## initialise components with fixed parameters
        self.source = flow.SourceM(name + '.source', media.Air(), flow_in, Xi_in, T_in, P_in)
        self.heater = heat.Heater(name + '.heater', media.Air(), T_out)
        self.sink = flow.Sink(name + '.sink', media.Air())

    def run_sol(self):
        self.connect_seq(self.source.F1, self.heater.F0) # feed Heater inlet flow
        self.heater.run_sol() # run internal solver in heater model
        self.connect_seq(self.heater.F1, self.sink.F0) # feed sink inlet flow

class SimpleCompressorSys(SysBase):
    def __init__(self, name, flow_in, Xi_in, T_in, P_in, P_out):
        self.__init__(locals())
        ## initialise components with fixed parameters
        self.source = flow.SourceM(name + '.source', media.Air(), flow_in, Xi_in, T_in, P_in)
        self.compressor = components.CompIsent(name + '.compressor', media.Air(), F1_P=P_out)
        self.sink = flow.Sink(name + '.sink', media.Air())

    def run_sol(self):
        self.connect_seq(self.source.F1, self.compressor.F0) # feed Compressor inlet flow
        self.compressor.run_sol() # run internal solver in compressor model
        self.connect_seq(self.compressor.F1, self.sink.F0) # feed sink inlet flow

class Ncomp(SysBase):
    '''
    N stages compressors with intermediate cooling. It creates a n-number of compressor objects 
    together with a cooler in front, a flange in (F0) and one out (F1).
    The efficiency of compression, the outlet pressure for each compressor and the outlet 
    temperature of each cooler must be given in a list of the N size. 
    IMPORTANT: the minimum number of stages is 2
    '''

    def __init__(self, name, Medium, n=1., Xi=None, T_in=400, P_in=1., N_stage=2, eff_comp=[0.7, 0.7], P_comp=[2., 3.], T_HT=[350., 350.]):  
        super().__init__(locals()) 
        assert N_stage >= 2, "Minimum 2 compression stages required for valid configuration"
        assert len(eff_comp) == N_stage, "eff_comp list length must match N_stage"
        assert len(P_comp) == N_stage, "P_comp list length must match N_stage"
        assert len(T_HT) == N_stage, "T_HT list length must match N_stage"
        self.N = N_stage

        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=n, Xi=Xi, T=T_in, P=P_in)  
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=n, Xi=Xi, T=298, P=1)

        self.compressors = []
        self.heaters = []

        for i in range(self.N):
            comp = components.CompIsent(f"{self.name}.comp{i+1}", Medium=Medium, eff=eff_comp[i], F1_P=P_comp[i])
            heater = heat.Heater(f"{self.name}.HT{i+1}", Medium0=Medium, Medium1=Medium, F1_T=T_HT[i], F1_P=P_comp[i])
            
            self.compressors.append(comp)
            self.heaters.append(heater)
            setattr(self, f"comp{i+1}", comp)
            setattr(self, f"HT{i+1}", heater)

        self.getObjs()

    def seq_sol(self, store=False):
        current_flow = self.F0
        for idx, (comp, heater) in enumerate(zip(self.compressors, self.heaters)):
            self.connect_seq(current_flow, comp.F0, store=store)
            comp.run_sol()
            
            self.connect_seq(comp.F1, heater.F0, store=store)
            heater.run_sol()
            
            current_flow = heater.F1
            if idx == self.N - 1:  # Final connection
                self.connect_seq(current_flow, self.F1, store=store)
    
    def performance_report(self):
        """Generate system performance summary"""
        return {
            "total_pressure_ratio": self.F1.P / self.F0.P,
            "stages": [
                {
                    "compressor": comp.name,
                    "pressure_ratio": comp.F1.P / comp.F0.P,
                    "efficiency": comp.eff,
                    "cooler_temp_drop": heater.F0.T - heater.F1.T
                }
                for comp, heater in zip(self.compressors, self.heaters)
            ]
        }

    def add_stage(self, eff, pressure, temp):
        """Dynamically add new compression stage"""
        self.N += 1
        new_comp = components.CompIsent(f"comp{self.N}", Medium=self.Medium, eff=eff, F1_P=pressure)
        new_heater = heat.Heater(f"HT{self.N}", Medium0=self.Medium, Medium1=self.Medium, F1_T=temp, F1_P=pressure)
        
        self.compressors.append(new_comp)
        self.heaters.append(new_heater)
    
    def run_sol(self):
        self.seq_sol()
 
