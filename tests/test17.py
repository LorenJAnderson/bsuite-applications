'Finding length of pandas dataframes'
import os
import pandas as pd

directory = '../results/PPO_tuned/'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # print(f)
    df = pd.read_csv(f)
    num_rows = df.shape[0]
    if num_rows != 49:
        print(f, num_rows)
