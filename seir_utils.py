# seir_utils.py

import pandas as pd
import numpy as np
import pints

def extract_initial_conditions_and_observed(filepath, time_col='time'):
    """
    Efficiently extract SEIR initial conditions and observed values from a CSV file.
    """
    usecols = [
        time_col,
        "InfectionStatus.Susceptible",
        "InfectionStatus.Exposed",
        "InfectionStatus.Recovered",
        "InfectionStatus.InfectASympt",
        "InfectionStatus.InfectMild",
        "InfectionStatus.InfectGP",
        "InfectionStatus.InfectHosp",
        "InfectionStatus.InfectICU",
        "InfectionStatus.InfectICURecov"
    ]

    # Load only necessary columns
    df = pd.read_csv(filepath, usecols=usecols)

    # Group and sum in one pass
    grouped = df.groupby(time_col, sort=False).sum(numeric_only=True)

    # Sum infected columns using .sum(axis=1) directly
    infected_cols = grouped.columns.difference(
        ["InfectionStatus.Susceptible", "InfectionStatus.Exposed", "InfectionStatus.Recovered"]
    )
    infected = grouped[infected_cols].sum(axis=1)

    # Assemble SEIR matrix (ensure column order)
    observed = np.stack([
        grouped["InfectionStatus.Susceptible"].values,
        grouped["InfectionStatus.Exposed"].values,
        infected.values,
        grouped["InfectionStatus.Recovered"].values
    ], axis=1)

    initial_conditions = observed[0].tolist()
    return initial_conditions, observed

def add_gaussian_noise(observed, sigma):
    noise = np.random.normal(loc=0, scale=sigma, size=observed.shape)
    noisy_data = observed + noise
    return np.clip(noisy_data, a_min=0, a_max=None)  # Ensure no negative populations