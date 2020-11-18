import numpy as np
import pandas as pd


class DataLoader:

    def __init__(self, filename):
        csv = pd.read_csv(filename, index_col=0)
        print(csv)
        ss = csv.iloc[:, -1].to_numpy()
        print(ss.dtype)

        narray = csv.to_numpy()
        ss1 = narray[:,-1]
        print(ss1.dtype)




if __name__ == "__main__":
    dl = DataLoader("datasets/fifa19/new_data.csv")
