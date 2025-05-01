import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv("DF_data_cleaned.csv")

# First split into train and temp (val+test)
df_train, df_test = train_test_split(df, test_size=0.15, shuffle=False)


sample_found = False
random_state = 0

while sample_found == False: 

    random_state += 1 

