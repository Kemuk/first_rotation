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
    def __init__(self, initial_conditions=None, real_Rt=0):
        super().__init__()
        self.initial_conditions = initial_conditions

        self.real_Rt=real_Rt
        self.R_t_estimate = 0
        self.name = None
        self.debug_history = []

        self.beta_estimate=None
        self.kappa_estimate=None
        self.gamma_estimate=None

        self.real_beta=None
        self.real_kappa=None
        self.real_gamma=None

    @abstractmethod
    def n_parameters(self):
        pass

    def n_outputs(self):
        return 4  # [S, E, I, R]

    @abstractmethod
    def _full_simulate(self, initial_conditions, parameters, times):
        pass

    def compare_estimates_to_truth(self):
        """
        Compare estimated parameters with their true values, including percentage error.
        """
        rows = []

        def add_row(param_name, est, true):
            error = abs(est - true)
            perc_error = (error / true * 100) if true != 0 else float('inf')
            rows.append({
                "Parameter": param_name,
                "Estimated": round(est, 6),
                "True": round(true, 6),
                "Error": round(error, 6),
                "Percentage Error": round(perc_error, 2)
            })

        add_row("beta", self.beta_estimate, self.real_beta)
        add_row("kappa", self.kappa_estimate, self.real_kappa)
        add_row("gamma", self.gamma_estimate, self.real_gamma)
        add_row("R_t", self.R_t_estimate, self.real_Rt)

        return pd.DataFrame(rows)



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
        
        self.beta_estimate=np.mean(beta_estimate)

        self.R_t_estimate = np.mean(crude_R_t[np.isfinite(crude_R_t)])

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
            # 2x2 grid for Simulated vs Observed
            fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            axs = axs.flatten()

            for i, ax in enumerate(axs):
                ax.plot(times, states[:, i], label="Simulated", linewidth=2)
                ax.plot(times, observed[:, i], label="Observed", linestyle='--')
                ax.set_ylabel(labels[i])
                ax.legend()
                ax.grid(True)
                ax.yaxis.set_major_formatter(formatter)

            for ax in axs[2:]:
                ax.set_xlabel("Time")
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
        """
        Fit the model parameters to observed data using PINTS optimizer.
        Stores fitted parameters as class attributes and returns key results.
        
        Args:
            times: Time points for simulation
            observed: Observed data to fit against
            x0: Initial parameter guess
            sigma: Standard deviation for likelihood function
            
        Returns:
            dict: Dictionary containing:
                - 'parameters': Fitted parameter values
                - 'simulated': Simulated output using fitted parameters
                - 'comparison': DataFrame comparing estimated vs. true parameters
        """
        self.set_true_rates_from_observed(observed, times)

        problem = pints.MultiOutputProblem(self, times, observed)
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma=np.ones(4) * sigma)

        lower, upper = self.default_bounds()
        prior = pints.UniformLogPrior(lower, upper)
        posterior = pints.LogPosterior(log_likelihood, prior)

        optimiser = pints.OptimisationController(posterior, x0, method=pints.XNES)
        optimiser.set_max_iterations(1000)
        optimiser.set_parallel(False)

        self.debug_history.clear()
        found_params, _ = optimiser.run()
        
        # Store original parameters as class attribute
        self.found_params = found_params
        
        # Get the simulation output with fitted parameters
        sim_output = self.simulate(found_params, times)
        
        # Call subclass implementation to update class attributes
        self.postprocess_fit_parameters(found_params)
        
        # Display the results
        self.plot(times, sim_output, title=f"Fitted {self.name}", observed=observed)
        if self.debug_history:
            self.plot_debug_history()

        # Get parameter comparison
        comparison_df = self.compare_estimates_to_truth()
        
        # Create the results dictionary with just the requested elements
        results = {
            'parameters': found_params,
            'simulated': sim_output,
            'comparison': comparison_df
        }
        
        # Print parameter comparison
        print("\nParameter Comparison:")
        print(comparison_df)
        
        return results
    def set_true_rates_from_observed(self, observed, times):
        """
        Estimate and set average true beta, gamma, and kappa from observed data.

        Args:
            observed (np.ndarray): Observed data of shape (n_timepoints, 4)
            times (np.ndarray): Time points associated with the observed data
        """
        dt = times[1] - times[0]
        S = observed[:, 0]
        E = observed[:, 1]
        I = observed[:, 2]
        R = observed[:, 3]
        N = S[0] + E[0] + I[0] + R[0]

        dS_dt = np.gradient(S, dt)
        dE_dt=np.gradient(E,dt)
        dI_dt = np.gradient(I, dt)
        dR_dt = np.gradient(R, dt)

        with np.errstate(divide='ignore', invalid='ignore'):
            # Avoid division by zero or negative inputs
            safe_SI = np.where((S * I) == 0, np.nan, S * I)
            beta = -dS_dt * N / safe_SI

            safe_I = np.where(I == 0, np.nan, I)
            gamma = dR_dt / safe_I

            # recompute with stable gamma (already has nans in unsafe places)
            safe_E = np.where(E == 0, np.nan, E)
            #kappa = (dI_dt + gamma * I) / safe_E
            kappa =  (beta/N * S * I -dE_dt )/ safe_E

        # Final assignment using nanmean
        self.real_beta = np.nanmean(beta)
        self.real_gamma = np.nanmean(gamma)
        self.real_kappa = np.nanmean(kappa)



    def ks_test_summary(self, observed, simulated, times=None, alpha=0.05):
        """
        Plot ECDFs with arrows showing KS statistics, and return KS stats in a DataFrame.

        Args:
            observed (np.ndarray): Observed data (T, 4).
            simulated (np.ndarray): Simulated model output (T, 4).
            times: Optional; not used.

        Returns:
            pd.DataFrame: KS statistics per compartment.
        """
        labels = ['Susceptible', 'Exposed', 'Infectious', 'Recovered']
        stats_data = []

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()

        for i, ax in enumerate(axs):
            obs_sorted = np.sort(observed[:, i])
            sim_sorted = np.sort(simulated[:, i])

            ecdf_obs = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
            ecdf_sim = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted)

            all_vals = np.union1d(obs_sorted, sim_sorted)
            ecdf_obs_interp = np.searchsorted(obs_sorted, all_vals, side='right') / len(obs_sorted)
            ecdf_sim_interp = np.searchsorted(sim_sorted, all_vals, side='right') / len(sim_sorted)

            diffs = np.abs(ecdf_obs_interp - ecdf_sim_interp)
            ks_stat = np.max(diffs)
            ks_index = np.argmax(diffs)
            x_ks = all_vals[ks_index]
            y1 = ecdf_obs_interp[ks_index]
            y2 = ecdf_sim_interp[ks_index]

            stats_data.append({
                "Compartment": labels[i],
                "KS Statistic": ks_stat
            })

            # Plot ECDFs
            ax.plot(obs_sorted, ecdf_obs, label='Observed', linestyle='--')
            ax.plot(sim_sorted, ecdf_sim, label='Simulated', linestyle='-')
            ax.set_title(f'ECDF – {labels[i]}')
            ax.grid(True)
            ax.legend()

            # Annotate KS arrow
            y_min, y_max = sorted([y1, y2])
            ax.annotate(
                "", xy=(x_ks, y_max), xytext=(x_ks, y_min),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5)
            )
            ax.text(
                x_ks, (y_min + y_max) / 2,
                f"KS={ks_stat:.3f}", color='red', fontsize=9,
                ha='left', va='center', rotation=90,
                bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2')
            )

        plt.tight_layout()
        plt.show()

        ks_df = pd.DataFrame(stats_data)
        print("\n=== KS Statistics ===")
        print(ks_df.to_string(index=False))

        return ks_df
    
class SimpleSEIRModel(AbstractSEIRModel):
    def __init__(self, initial_conditions, real_Rt):
        super().__init__(initial_conditions, real_Rt)
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
        return [0.05, 0.05, 0.05], [5.0, 1.0, 1.0]

    def postprocess_fit_parameters(self, params):
        """
        Store fitted parameters as class attributes.
        
        Args:
            params: The fitted parameters [beta, kappa, gamma]
        """
        beta, kappa, gamma = params
        self.beta_estimate = beta
        self.kappa_estimate = kappa  
        self.gamma_estimate = gamma
    

class RocheModel(AbstractSEIRModel):
    def __init__(self, initial_conditions, real_Rt):
        super().__init__(initial_conditions, real_Rt)
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
        """
        Calculate and store derived parameters from the fitted Roche model parameters.
        
        Args:
            params: The fitted parameters [C, beta_min, beta_max, stringency50, k, k_s, k_ri]
        """
        C, beta_min, beta_max, stringency50, k, k_s, k_ri = params

        # Store original parameters
        self.C_estimate = C
        self.beta_min_estimate = beta_min
        self.beta_max_estimate = beta_max
        self.stringency50_estimate = stringency50
        self.k_estimate = k
        self.k_s_estimate = k_s
        self.k_ri_estimate = k_ri

        # Calculate derived parameters
        kappa = 1 / k
        gamma = 1 / (k_s + k_ri)

        self.kappa_estimate = kappa
        self.gamma_estimate = gamma

        # Print optimised parameters as a DataFrame
        param_df = pd.DataFrame([{
            "C": C,
            "beta_min": beta_min,
            "beta_max": beta_max,
            "stringency50": stringency50,
            "k": k,
            "k_s": k_s,
            "k_ri": k_ri
        }])
        print("\n=== Optimised Parameters ===")
        print(param_df.to_string(index=False))