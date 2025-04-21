import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv('metrics.csv')


y = df['covh2dtime']
X = df[['covh2dbytes']] 

print(y)
print(X) 

plt.figure(figsize=(10,6))
plt.scatter(df['covh2dbytes'],y,color='blue',label='Data points')
plt.title('Covaraiance H2D Time vs Covariance H2D  Bytes')
plt.xlabel('covh2dbytes')
plt.ylabel('covh2dtime')
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

y = df['covh2dtime']
X = df[['covh2dbytes']] 
model = sm.OLS(y,X).fit()
summary=model.summary()
print(summary)