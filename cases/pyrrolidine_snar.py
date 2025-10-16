import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
from scipy.stats.qmc import LatinHypercube

from .benchmark import Benchmark


class PyrrolidineSNAr(Benchmark):
    """Benchmark representing a nucleophilic aromatic substitution (SNAr) reaction.
    Pyrrolidine is the nucleophile to attack the electron-deficient aromatic ring.

    The SNAr reaction occurs in a plug flow reactor where residence time, nucleophile 
    concentration, and temperature can be adjusted.

    Parameters
    ----------
    noise_level: float, optional
        The mean of the random noise added to the concentration measurements in terms of
        percent of the signal. Default is 0.

    random_seed: int, optional
        The Random seed to generate noises. Default is 0.

    Notes
    -----
    This benchmark relies on the kinetics observerd by [Hone] et al. The mechanistic 
    model is integrated using scipy to find outlet concentrations of all species.

    Parameter units :
        - 'Radius'                  : m
        - 'Length'                  : m
        - 'Activation_Energy'       : kJ/mol
        - 'Pre-exponential_Factor'  : 
        - 'Concentration'           : mol/L
        - 'Temperature'             : oC
        - 'Residence Time'          : min

    References
    ----------
    .. [Hone] C. A. Hone et al., React. Chem. Eng., 2017, 2, 103–108. DOI:
       `10.1039/C6RE00109B <https://doi.org/10.1039/C6RE00109B>`_

    """

    species = ["dfnb", "prld", "ortho", "para", "bis"]
    reactions = [
        "dfnb + prld > ortho",
        "dfnb + prld > para",
        "ortho + prld > bis",
        "para + prld > bis",
    ]
    streams = ["tubular_flow"]

    def __init__(self, phenos, noise_level=0, random_seed=0):
        structure_params = self._setup_structure_params()
        physics_params = self._setup_physics_params()
        kinetics_params = self._setup_kinetics_params()
        transport_params = self._setup_transport_params()
        operation_params = self._setup_operation_params()
        operation_name2ind = self._setup_operation_name2ind()
        measure_ind2name = self._setup_measure_ind2name()
        var2unit = self._setup_var2unit()
        super().__init__(
            structure_params,
            physics_params,
            kinetics_params,
            transport_params,
            operation_params,
            operation_name2ind,
            measure_ind2name,
            var2unit
        )
        self._validate_phenos(phenos)
        self.phenos = phenos
        self.noise_level = noise_level
        self.random_seed = random_seed
    
    def _validate_phenos(self, phenos):
        assert isinstance(phenos, dict), "phenos should be a dictionary"
        assert "Mass accumulation" in phenos and phenos["Mass accumulation"] == "Continuous", \
            "SNAr reaction is operated in a 'Continuous' flow reactor"
        assert "Flow pattern" in phenos and phenos["Flow pattern"] == "Tubular_Flow", \
            "SNAr reaction is operated with 'Tubular Flow'"
        assert "Mass transport" in phenos and phenos["Mass transport"] == [], \
            "The tubular flow is well mixed along the radical direction"
        assert "Mass equilibrium" in phenos and phenos["Mass equilibrium"] == [], \
            "The tubular flow is well mixed along the radical direction"

    def _setup_structure_params(self):
        structure_params = {}
        return structure_params

    def _setup_physics_params(self):
        physics_params = {}
        return physics_params

    def _setup_kinetics_params(self):
        kinetics_params = {
            ("Activation_Energy", None, 0, 0, None): 33.3,  # kJ/mol
            ("Activation_Energy", None, 0, 1, None): 35.3,  # kJ/mol
            ("Activation_Energy", None, 0, 2, None): 38.9,  # kJ/mol
            ("Activation_Energy", None, 0, 3, None): 44.8,  # kJ/mol
            ("Pre-exponential_Factor", None, 0, 0, None): 0.57900,  # L/mol s
            ("Pre-exponential_Factor", None, 0, 1, None): 0.02700,  # L/mol s
            ("Pre-exponential_Factor", None, 0, 2, None): 0.00865,  # L/mol s
            ("Pre-exponential_Factor", None, 0, 3, None): 0.01630,  # L/mol s
            ("Stoichiometric_Coefficient", None, None, 0, 0): -1,
            ("Stoichiometric_Coefficient", None, None, 0, 1): -1,
            ("Stoichiometric_Coefficient", None, None, 0, 2): 1,
            ("Stoichiometric_Coefficient", None, None, 1, 0): -1,
            ("Stoichiometric_Coefficient", None, None, 1, 1): -1,
            ("Stoichiometric_Coefficient", None, None, 1, 3): 1,
            ("Stoichiometric_Coefficient", None, None, 2, 1): -1,
            ("Stoichiometric_Coefficient", None, None, 2, 2): -1,
            ("Stoichiometric_Coefficient", None, None, 2, 4): 1,
            ("Stoichiometric_Coefficient", None, None, 3, 1): -1,
            ("Stoichiometric_Coefficient", None, None, 3, 3): -1,
            ("Stoichiometric_Coefficient", None, None, 3, 4): 1,
            ("Partial_Order", None, None, 0, 0): 1,
            ("Partial_Order", None, None, 0, 1): 1,
            ("Partial_Order", None, None, 1, 0): 1,
            ("Partial_Order", None, None, 1, 1): 1,
            ("Partial_Order", None, None, 2, 1): 1,
            ("Partial_Order", None, None, 2, 2): 1,
            ("Partial_Order", None, None, 3, 1): 1,
            ("Partial_Order", None, None, 3, 3): 1,
        }
        return kinetics_params

    def _setup_transport_params(self):
        transport_params = {}
        return transport_params

    def _setup_operation_params(self):
        operation_params = {
            ("Concentration", None, 0, None, 0): 0.2,  # mol/L
            ("Concentration", None, 0, None, 1): None,  # mol/L
            ("Temperature", None, None, None, None): None,  # oC
            ("Residence_Time", None, None, None, None): None,  # min
        }
        return operation_params

    def _setup_operation_name2ind(self):
        operation_name2ind = {
            "prld_conc": ("Concentration", None, 0, None, 1),
            "temp": ("Temperature", None, None, None, None),
            "t_r": ("Residence_Time", None, None, None, None),
        }
        return operation_name2ind

    def _setup_measure_ind2name(self):
        measure_ind2name = {
            ("Concentration", None, 0, None, 2): "outlet_ortho_conc",
        }
        return measure_ind2name

    def _setup_var2unit(self):
        var2unit = {
            "Activation_Energy": "kJ/mol",
            "Pre-exponential_Factor": None,
            "Temperature": "oC",
            "Concentration": "mol/L",
            "Residence_Time": "min",
        }
        return var2unit

    def _simulate(self, params):
        R = 8.314
        t_r = params[("Residence_Time", None, None, None, None)]
        t_r *= 60
        T = params[("Temperature", None, None, None, None)]
        T += 273.15
        c_0 = np.zeros((1, 5), dtype=np.float64)
        c_0[0, 0] = params[("Concentration", None, 0, None, 0)]
        c_0[0, 1] = params[("Concentration", None, 0, None, 1)]
        nu = np.zeros((4, 5), dtype=np.float64)
        nu[0, 0] = params[("Stoichiometric_Coefficient", None, None, 0, 0)]
        nu[0, 1] = params[("Stoichiometric_Coefficient", None, None, 0, 1)]
        nu[0, 2] = params[("Stoichiometric_Coefficient", None, None, 0, 2)]
        nu[1, 0] = params[("Stoichiometric_Coefficient", None, None, 1, 0)]
        nu[1, 1] = params[("Stoichiometric_Coefficient", None, None, 1, 1)]
        nu[1, 3] = params[("Stoichiometric_Coefficient", None, None, 1, 3)]
        nu[2, 1] = params[("Stoichiometric_Coefficient", None, None, 2, 1)]
        nu[2, 2] = params[("Stoichiometric_Coefficient", None, None, 2, 2)]
        nu[2, 4] = params[("Stoichiometric_Coefficient", None, None, 2, 4)]
        nu[3, 1] = params[("Stoichiometric_Coefficient", None, None, 3, 1)]
        nu[3, 3] = params[("Stoichiometric_Coefficient", None, None, 3, 3)]
        nu[3, 4] = params[("Stoichiometric_Coefficient", None, None, 3, 4)]
        n = np.zeros((4, 5), dtype=np.float64)
        n[0, 0] = params[("Partial_Order", None, None, 0, 0)]
        n[0, 1] = params[("Partial_Order", None, None, 0, 1)]
        n[1, 0] = params[("Partial_Order", None, None, 1, 0)]
        n[1, 1] = params[("Partial_Order", None, None, 1, 1)]
        n[2, 1] = params[("Partial_Order", None, None, 2, 1)]
        n[2, 2] = params[("Partial_Order", None, None, 2, 2)]
        n[3, 1] = params[("Partial_Order", None, None, 3, 1)]
        n[3, 3] = params[("Partial_Order", None, None, 3, 3)]
        A = np.zeros((1, 4), dtype=np.float64)
        A[0, 0] = params[("Pre-exponential_Factor", None, 0, 0, None)]
        A[0, 1] = params[("Pre-exponential_Factor", None, 0, 1, None)]
        A[0, 2] = params[("Pre-exponential_Factor", None, 0, 2, None)]
        A[0, 3] = params[("Pre-exponential_Factor", None, 0, 3, None)]
        E_a = np.zeros((1, 4), dtype=np.float64)
        E_a[0, 0] = params[("Activation_Energy", None, 0, 0, None)]
        E_a[0, 1] = params[("Activation_Energy", None, 0, 1, None)]
        E_a[0, 2] = params[("Activation_Energy", None, 0, 2, None)]
        E_a[0, 3] = params[("Activation_Energy", None, 0, 3, None)]
        E_a *= 1000

        def _derivative(t, c):
            c = c.reshape((1, 5))
            r_r = np.zeros((1, 4), dtype=np.float64)
            r_r[0, 0] = A[0, 0] * np.exp(-E_a[0, 0] / R * (1 / T - 1 / 363.15)) * np.prod(c[0] ** n[0])
            r_r[0, 1] = A[0, 1] * np.exp(-E_a[0, 1] / R * (1 / T - 1 / 363.15)) * np.prod(c[0] ** n[1])
            r_r[0, 2] = A[0, 2] * np.exp(-E_a[0, 2] / R * (1 / T - 1 / 363.15)) * np.prod(c[0] ** n[2])
            r_r[0, 3] = A[0, 3] * np.exp(-E_a[0, 3] / R * (1 / T - 1 / 363.15)) * np.prod(c[0] ** n[3])
            dc_dt = np.matmul(r_r, nu)
            return dc_dt

        t_eval = np.linspace(0, t_r, 201)
        res = solve_ivp(_derivative, (0, t_r), c_0.reshape(-1, ),
                        method="LSODA", t_eval=t_eval, atol=1e-8)
        t_eval /= 60
        return t_eval, res.y

    def _run(self, operation_params, kinetics_params, transport_params):
        assert set(operation_params.keys()).issubset(
            set(self.operation_params().keys())), "Unknown operation parameters included"
        assert set(kinetics_params.keys()).issubset(
            set(self.kinetics_params().keys())), "Unknown kinetics parameters included"
        assert set(transport_params.keys()).issubset(
            set(self.transport_params().keys())), "Unknown mole transport parameters included"
        params = self.params()
        params.update(operation_params)
        if kinetics_params is not None:
            params.update(kinetics_params)
        if transport_params is not None:
            params.update(transport_params)
        return self._simulate(params)

    def run(self, operation_params, kinetics_params=None, transport_params=None):
        if kinetics_params is None:
            kinetics_params = self.kinetics_params()
        if transport_params is None:
            transport_params = self.transport_params()
        t, cs = self._run(operation_params, kinetics_params, transport_params)
        return t, cs

    def run_dataset(self, dataset):
        dataset = dataset.copy()
        res = {name: [] for name in self._measure_ind2name.values()}
        for i in range(len(dataset)):
            operation_params = {}
            for name, ind in self._operation_name2ind.items():
                operation_params[ind] = dataset.loc[i, name]
            _, cs = self.run(operation_params)
            for ind, name in self._measure_ind2name.items():
                res[name].append(cs[ind[-1], -1])
        for name, val in res.items():
            dataset[name] = val
        return dataset

    def generate_lhs_dataset(self, operation_param_ranges, num_points):
        lhs = LatinHypercube(len(operation_param_ranges), rng=self.random_seed)
        data = lhs.random(num_points)
        for i, (_, param_range) in enumerate(operation_param_ranges.items()):
            data[:, i] = data[:, i] * (param_range[1] - param_range[0]) + param_range[0]
        dataset = pd.DataFrame(data, columns=list(operation_param_ranges.keys()))
        return self.run_dataset(dataset)

    def calibrate(self, cal_param_bounds, dataset):
        params = self.params()
        def calc_mse(p):
            mse = 0
            cal_params = {cal_param_ind: _p for cal_param_ind, _p in zip(cal_param_bounds.keys(), p)}
            params.update(cal_params)
            for i in range(len(dataset)):
                operation_params = {ind: dataset.loc[i, name] for name, ind in self._operation_name2ind.items()}
                params.update(operation_params)
                t, cs = self._simulate(params)
                for ind, name in self._measure_ind2name.items():
                    mse += (cs[ind[-1], -1] - dataset.loc[i, name])**2
            return mse
        res = minimize(
            fun=calc_mse, 
            x0=[np.mean(v).item() for v in cal_param_bounds.values()], 
            method='L-BFGS-B',
            bounds=list(cal_param_bounds.values()),
        )
        cal_params = {ind: round(v.item(), 6) for ind, v in zip(cal_param_bounds.keys(), res.x)}
        return cal_params


if __name__ == "__main__":
    pyrrolidine_snar = PyrrolidineSNAr()
    operation_params = {
        ("Residence_Time", None, None, None, None): 1,
        ("Temperature", None, None, None, None): 100,
        ("Concentration", None, 0, None, 1): 0.4,
    }
    print(pyrrolidine_snar.run(operation_params))