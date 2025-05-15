from abc import ABC, abstractmethod
import numpy as np
import pints
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d
np.random.seed(42)

class AbstractSEIRModel(pints.ForwardModel, ABC):
    def __init__(self, initial_conditions=None):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.crude_R_t = 0
        self.name = None

    @abstractmethod
    def n_parameters(self):
        pass

    def n_outputs(self):
        return 4  # [S, E, I, R]

    @abstractmethod
    def _full_simulate(self, initial_conditions, parameters, times):
        pass

    def simulate(self, parameters, times):
        if self.initial_conditions is None:
            raise ValueError("Initial conditions must be set before simulation.")
        states = self._full_simulate(self.initial_conditions, parameters, times)
        self._estimate_crude_R_t(states)
        return states

    def _estimate_crude_R_t(self, states):
        pop_size = np.sum(states[0])
        susceptible = states[:, 0]
        infected = states[:, 2]
        recovered = states[:, 3]

        dS_dt = np.gradient(susceptible)
        dR_dt = np.gradient(recovered)
        with np.errstate(divide='ignore', invalid='ignore'):
            beta_estimate = -dS_dt * pop_size / (susceptible * infected)
            gamma_estimate = dR_dt / infected
            crude_R_t = beta_estimate * susceptible / (pop_size * gamma_estimate)

        self.crude_R_t = np.mean(crude_R_t[np.isfinite(crude_R_t)])

    def plot(self, times, states, title="SEIR Simulation", observed=None):
        labels = ['Susceptible', 'Exposed', 'Infectious', 'Recovered']

        def thousands_formatter(x, pos):
            return f'{int(x):,}'

        formatter = FuncFormatter(thousands_formatter)

        if observed is not None:
            # Plot with 4 subplots for Simulated vs Observed
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(times, states[:, i], label="Simulated", linewidth=2)
                ax.plot(times, observed[:, i], label="Observed", linestyle='--')
                ax.set_ylabel(labels[i])
                ax.legend()
                ax.grid(True)
                ax.yaxis.set_major_formatter(formatter)
            axs[-1].set_xlabel("Time")
        else:
            # Single combined plot
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(4):
                ax.plot(times, states[:, i], label=labels[i], linewidth=2)
            ax.set_xlabel("Time")
            ax.set_ylabel("Population")
            ax.legend()
            ax.grid(True)
            ax.yaxis.set_major_formatter(formatter)

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    @abstractmethod
    def default_bounds(self):
        """
        Return (lower, upper) bounds for model parameters.
        """
        pass

    def postprocess_fit_parameters(self, parameters):
        """
        Optionally overridden by subclasses to compute beta, gamma, kappa.
        """
        return {}

    def fit_with_pints(self, times, observed, x0, sigma=10.0):
        problem = pints.MultiOutputProblem(self, times, observed)
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma=np.ones(4) * sigma)

        lower, upper = self.default_bounds()
        prior = pints.UniformLogPrior(lower, upper)
        posterior = pints.LogPosterior(log_likelihood, prior)

        optimiser = pints.OptimisationController(posterior, x0, method=pints.XNES)
        optimiser.set_max_iterations(1000)
        optimiser.set_parallel(False)

        found_params, found_value = optimiser.run()
        sim_output = self.simulate(found_params, times)

        # Plot comparison
        self.plot(times, sim_output, title=f"Fitted {self.name}", observed=observed)

        return {
            "optimized_parameters": found_params,
            "log_posterior": found_value,
            "R_estimate": self.crude_R_t,
            **self.postprocess_fit_parameters(found_params),
        }
    
class SimpleSEIRModel(AbstractSEIRModel):
    def __init__(self, initial_conditions):
        super().__init__(initial_conditions)
        self.name="SimpleSEIRModel"

    def n_parameters(self):
        return 3

    def _full_simulate(self, initial_conditions, parameters, times):
        beta, kappa, gamma = parameters
        dt = times[1] - times[0]
        num_steps = len(times)

        S0, E0, I0, R0 = initial_conditions
        states = np.zeros((num_steps, 4))
        states[0] = [S0, E0, I0, R0]

        for i in range(1, num_steps):
            S, E, I, R = states[i - 1]
            N = S + E + I + R
            dS = -beta * S * I / N
            dE = beta * S * I / N - kappa * E
            dI = kappa * E - gamma * I
            dR = gamma * I
            states[i] = [S + dt * dS, E + dt * dE, I + dt * dI, R + dt * dR]

        return states

    def default_bounds(self):
        return [0.1, 0.05, 0.05], [5.0, 1.0, 1.0]

    def postprocess_fit_parameters(self, params):
        beta, kappa, gamma = params
        return {
            "beta": beta,
            "kappa": kappa,
            "gamma": gamma
        }
  

class RocheModel(AbstractSEIRModel):
    def __init__(self, initial_conditions):
        super().__init__(initial_conditions)
        self.name="Roche Model"

    def n_parameters(self):
        return 8

    def _full_simulate(self, initial_conditions, parameters, times):
        C, beta_min, beta_max, stringency, stringency50, k, k_s, k_ri = parameters

        dt = times[1] - times[0]
        num_steps = len(times)

        # Initial conditions: [S, E, I, R]
        S0, E0, I0, R0 = initial_conditions
        states = np.zeros((num_steps, 4))
        states[0] = [S0, E0, I0, R0]

        #symptomatic transmission rate βs

        N = S0 + E0 + I0 + R0

        kappa = 1/k

        gamma = 1/(k_s+k_ri)
        
        beta_s= (beta_max - (beta_max-beta_min) * ((stringency**gamma)/(stringency**gamma + stringency50**gamma)))/2*N

        beta=C*(beta_s/(2*N))

        for i in range(1, num_steps):
            S, E, I, R = states[i - 1]
            

            
            dS = -beta * S * I / N
            dE = beta * S * I / N - kappa * E
            dI = kappa * E - gamma * I
            dR = gamma * I

            states[i] = [S + dt * dS, E + dt * dE, I + dt * dI, R + dt * dR]

        return states

    def default_bounds(self):
        lower = [1, 0.01, 0.5, 0, 10, 1, 1, 1]
        upper = [100, 0.5, 2.0, 100, 100, 10, 14, 21]
        return lower, upper

    def postprocess_fit_parameters(self, params):
        C, beta_min, beta_max, stringency, stringency50, k, k_s, k_ri = params
        pop_size = sum(self.initial_conditions)
        kappa = 1 / k
        gamma = 1 / (k_s + k_ri)
        beta_s = (beta_max - (beta_max - beta_min) * ((stringency ** gamma) / (stringency ** gamma + stringency50 ** gamma))) / (2 * pop_size)
        beta = C * beta_s

        return {
            "beta": beta,
            "kappa": kappa,
            "gamma": gamma
        }
