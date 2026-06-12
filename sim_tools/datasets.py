"""
Datasets module

Contains functions for loading example data to demonstrate sim-tools
functionality.
"""

from pathlib import Path

import pandas as pd

FILE_NAME_NSPP_1 = "nspp_example1.csv"
PATH_NSPP_1 = Path(__file__).parent.joinpath("data", FILE_NAME_NSPP_1)


def load_banks_et_al_nspp() -> pd.DataFrame:
    """
    Load example Non-stationary poisson process data from Banks et al.

    The function reads in the mean inter-arrival times by interval.
    The arrival rate (1/mean_iat) is calculated.

    Returns
    -------
    pandas.DataFrame
    """
    arrivals = pd.read_csv(PATH_NSPP_1)
    arrivals["arrival_rate"] = 1 / arrivals["mean_iat"]
    return arrivals
