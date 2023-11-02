import pandas as pd

class Read:
    def __init__(self, path, columns:list):
        self.path = path
        self.columns = columns

    def read(self):
        df = pd.read_excel(self.path)
        df =  df[self.columns]
    
        # Select features (X) and target variable (y)
        X = df.drop(columns=['Nüks'])
        y = df['Nüks']

        return X, y
