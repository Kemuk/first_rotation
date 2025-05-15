from abc import ABC, abstractmethod
import numpy as np
import pints
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d
import math
from scipy.stats import ks_2samp
import pandas as pd
np.random.seed(42)

class AbstractSEIRModel(pints.ForwardModel, ABC):
    def __init__(self, initial_conditions=None):
        super().__init__()
        self.initial_conditions = initial_conditions
        self.crude_R_t = 0
        self.name = None
        self.debug_history = []

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

        # Allow debug info collection (if supported by subclass)
        try:
            states, debug_info = self._full_simulate(self.initial_conditions, parameters, times, return_debug=True)
            self.debug_history.append(debug_info)
        except TypeError:
            # Subclass didn't implement `return_debug`, fallback to old call
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

    def plot_debug_history(self):
        """
        Plot each parameter from debug_info over optimisation iterations
        in its own subplot.
        """
        if not self.debug_history:
            print("No debug history to plot.")
            return

        # Get all keys in debug_info
        keys = self.debug_history[0].keys()
        n_keys = len(keys)

        # Determine grid size
        n_cols = 3
        n_rows = math.ceil(n_keys / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, key in enumerate(keys):
            values = [d[key] for d in self.debug_history if key in d]
            axes[i].plot(values, label=key)
            axes[i].set_title(key)
            axes[i].set_xlabel("Optimisation iteration")
            axes[i].set_ylabel("Value")
            axes[i].grid(True)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Debug Parameters Over Optimisation", fontsize=16)
        plt.tight_layout()
        plt.show()

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

        optimiser = pints.OptimisationController(posterior, x0, method=pints.CMAES)
        optimiser.set_max_iterations(1000)
        optimiser.set_parallel(False)

        # Clear previous debug history
        self.debug_history.clear()

        found_params, found_value = optimiser.run()
        sim_output = self.simulate(found_params, times)

        # Plot simulation vs observed
        self.plot(times, sim_output, title=f"Fitted {self.name}", observed=observed)

        # Optional: plot debug info evolution
        if self.debug_history:
            self.plot_debug_history()

        return {
            "optimized_parameters": found_params,
            "log_posterior": found_value,
            "R_estimate": self.crude_R_t,
            **self.postprocess_fit_parameters(found_params),
            "results": sim_output
        }
    def ks_test_summary(self, observed, simulated, times=None, alpha=0.05):
        """
        Performs the Kolmogorov–Smirnov test comparing each compartment's simulated and observed data.

        H0: The simulated and observed data come from the same distribution.
        H1: The simulated and observed data come from different distributions.

        Args:
            observed (np.ndarray): Observed data of shape (n_timepoints, 4)
            simulated (np.ndarray): Simulated model output of shape (n_timepoints, 4)
            times (np.ndarray, optional): Time points (not used in test)
            alpha (float): Significance level for hypothesis testing (default 0.05)
        """
        labels = ['Susceptible', 'Exposed', 'Infectious', 'Recovered']
        ks_results = []

        for i, label in enumerate(labels):
            ks_stat, p_value = ks_2samp(observed[:, i], simulated[:, i])
            reject_null = p_value < alpha
            ks_results.append((label, ks_stat, p_value, reject_null))

        # Display results in a table
        ks_df = pd.DataFrame(
            ks_results,
            columns=['Compartment', 'KS Statistic', 'p-value', f'Reject H₀ (α={alpha})']
        )

        print("\n=== Kolmogorov–Smirnov Test Summary ===")
        print("H₀: Simulated and observed data come from the SAME distribution.")
        print("H₁: Simulated and observed data come from DIFFERENT distributions.")
        print(ks_df.to_string(index=False))

        # Plot empirical CDFs
        fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
        for i, ax in enumerate(axs):
            sorted_obs = np.sort(observed[:, i])
            sorted_sim = np.sort(simulated[:, i])
            ecdf_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)
            ecdf_sim = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)

            ax.plot(sorted_obs, ecdf_obs, label='Observed', linestyle='--')
            ax.plot(sorted_sim, ecdf_sim, label='Simulated', linestyle='-')
            ax.set_title(f'Empirical CDF - {labels[i]}')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    
class SimpleSEIRModel(AbstractSEIRModel):
    def __init__(self, initial_conditions):
        super().__init__(initial_conditions)
        self.name="SimpleSEIRModel"

    def n_parameters(self):
        return 3

    def _full_simulate(self, initial_conditions, parameters, times,  return_debug=False):
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

                
        debug_info = {
            "beta": beta,
            "gamma": gamma,
            "kappa": kappa
        }

        if return_debug:
            return states, debug_info
        
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
        self.stringency=1

    def n_parameters(self):
        return 7


    def _full_simulate(self, initial_conditions, parameters, times, return_debug=False):
        C, beta_min, beta_max, stringency50, k, k_s, k_ri = parameters
        stringency = self.stringency

        dt = times[1] - times[0]
        num_steps = len(times)

        S0, E0, I0, R0 = initial_conditions
        states = np.zeros((num_steps, 4))
        states[0] = [S0, E0, I0, R0]

        N = S0 + E0 + I0 + R0

        # Epidemiological rates
        kappa = 1 / k
        gamma = 1 / (k_s + k_ri)

        # Debug dictionary to store intermediate beta calculation values

        # Step 2: Compute sigmoid components
        theta_gamma = stringency ** gamma
        theta50_gamma = stringency50 ** gamma
        fraction = theta_gamma / (theta_gamma + theta50_gamma)

        # Step 3: Compute beta_s
        beta_s = (beta_max - (beta_max - beta_min) * fraction)

        # Step 4: Final beta
        beta = C * (beta_s /(2*N))

        ds_over_time = []
        de_over_time = []

        
        debug_info = {
            "C": C,
            "beta_min": beta_min,
            "beta_max": beta_max,
            "stringency": stringency,
            "stringency50": stringency50,
            "gamma": gamma,
            "theta_gamma": theta_gamma,
            "theta50_gamma": theta50_gamma,
            "fraction": fraction,
            "beta_s": beta_s,
            "beta": beta,
        }

        # SEIR integration
        for i in range(1, num_steps):
            S, E, I, R = states[i - 1]

            dS = -beta * S * I
            dE = beta * S * I- kappa * E
            dI = kappa * E - gamma * I
            dR = gamma * I

            ds_over_time.append(dS)
            de_over_time.append(dE)


            states[i] = [S + dt * dS, E + dt * dE, I + dt * dI, R + dt * dR]
        
        if return_debug:
            debug_info.update({
            "ds_over_time": np.array(ds_over_time),
            "de_over_time": np.array(de_over_time)
            })
            return states, debug_info

        return states



    def default_bounds(self):
        """
        Returns the default lower and upper bounds for the RocheModel parameters.

        These bounds are used by the optimiser to constrain the search space during
        parameter fitting. The parameters correspond to:

            1. C            - Scaling constant for transmission
            2. beta_min     - Minimum transmission rate
            3. beta_max     - Maximum transmission rate
            5. stringency50 - Stringency at which transmission is halved
            6. k            - Average latent period (days)
            7. k_s          - Average symptomatic infectious period (days)
            8. k_ri         - Average recovery or isolation period (days)

        Returns:
            tuple: A pair of lists (lower_bounds, upper_bounds) containing the 
                minimum and maximum allowable values for each parameter.
        """
        lower = [0.01, 0.01, 0.5, 1, 1, 1, 1]
        upper = [1, 0.5, 2.0, 100, 10, 14, 21]
        return lower, upper

    def postprocess_fit_parameters(self, params):
        C, beta_min, beta_max, stringency50, k, k_s, k_ri = params
        stringency=self.stringency
        pop_size = sum(self.initial_conditions)

        kappa = 1 / k
        gamma = 1 / (k_s + k_ri)

        theta_gamma = stringency ** gamma
        theta50_gamma = stringency50 ** gamma

        beta_s = beta_max - (beta_max - beta_min) * (theta_gamma / (theta_gamma + theta50_gamma))
        beta = C * (beta_s / (2 * pop_size))

        return {
            "beta": beta,
            "kappa": kappa,
            "gamma": gamma
        }
