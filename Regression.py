import pandas as pd
import quandl,math
import numpy as np
from sklearn import preprocessing ,cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime,arrow
import time
import pickle

style.use('ggplot')

#geting Data
df = quandl.get('WIKI/GOOGL')
#geting  Specific Data
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#creating a new column with specific calcuations
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'] )/ df['Adj. Close']*100
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open'] )/ df['Adj. Open']*100

#Setting Values  price  X         X             X
df = df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']] #features

#variable
forecast_col = 'Adj. Close'
#Fill data
df.fillna(-99999,inplace=True)

#predict 10% of data set
forecast_out = int(math.ceil(0.1*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label','Adj. Close'],1)) #features
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])  #labels

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2) 

clf = LinearRegression(n_jobs=-1) #classifier
#clf = svm.SVR()
clf.fit(X_train,y_train)
#creating a pickle To store the clasifier
with open('linearregression.pickle','wb') as f:
     pickle.dump(clf,f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test) #calculations

#print(accuracy)
forecast_set = clf.predict(X_lately)

print(forecast_set,accuracy,forecast_out)
df['Forecast'] = np.nan
#dates

#not Sure
#utc = arrow.utcnow()
#local = utc.to('US/Pacific')
last_date = df.iloc[-1].name
last_unix = time.time()
#last_unix = last_date.Timestamp() #correct line
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
       next_date = datetime.datetime.fromtimestamp(next_unix)
       next_unix += one_day
       df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()