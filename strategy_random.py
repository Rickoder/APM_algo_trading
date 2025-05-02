import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from useful_function import get_sr        

# Random signal
def random_signal(df):
    """
    Generate a random signal for testing purposes.
    """
    # Generate a random signal with values -1, 0, or 1
    # Here we create a 100x100 array of random signals
    rows = df.shape[0]

    rndm_signal = np.random.choice([-1, 0, 1], size=(rows)) 
    rndm_signal = pd.Series(rndm_signal)

    return rndm_signal