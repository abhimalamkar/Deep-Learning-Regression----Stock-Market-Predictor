import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd

df = pd.read_csv('train.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['pass'],1))
y = np.array(df['pass'])

print(df,X,y)