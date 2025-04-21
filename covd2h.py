import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv('metrics.csv')


y = df['covd2htime']
X = df[['covd2hbytes']] 

print(y)
print(X) 

plt.figure(figsize=(10,6))
plt.scatter(df['covd2hbytes'],y,color='blue',label='Data points')
plt.title('Covaraiance D2H Time vs Covariance D2H  Bytes')
plt.xlabel('covd2hbytes')
plt.ylabel('covd2htime')
plt.legend()
plt.grid(True)
plt.show()

plt.show()


model=LinearRegression()
model.fit(X,y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


featureval = np.array([[53]])
print(model.predict(featureval))

y = df['covd2htime']
X = df[['covd2hbytes']] 
model = sm.OLS(y,X).fit()
summary=model.summary()
print(summary)