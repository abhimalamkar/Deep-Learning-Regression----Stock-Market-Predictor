import pandas as pd
import quandl,math
import numpy as np
from sklearn import preprocessing ,cross_validation, svm
from sklearn.linear_model import LinearRegression

#geting Data
df = quandl.get('WIKI/GOOGL')
#geting  Specific Data
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#creating a new column with specific calcuations
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'] )/ df['Adj. Close']*100
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open'] )/ df['Adj. Open']*100
#Setting Values
df = df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']] #features

#variable
forecast_col = 'Adj. Close'
#Fill data
df.fillna(-99999,inplace=True)

#predict 10% of data set
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

X = np.array(df.drop(['label'],1)) #features
y = np.array(df['label'])  #labels

x = preprocessing.scale(x)