import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv('PA4training(1).csv')
validation = pd.read_csv("PA4validation.csv")


y = df['h2dtime_img_transfer']
X = df[['H2dsize']] 

print(y)
print(X) 

plt.figure(figsize=(10,6))
plt.scatter(df['H2dsize'],y,color='blue',label='Data points')
plt.title('Convolution H2D Time vs Convolution H2D Bytes')
plt.xlabel('convh2dbytes')
plt.ylabel('convh2dtime')
plt.legend()
plt.grid(True)
plt.show()

plt.show()


model=LinearRegression()
model.fit(X,y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

test = []

for image_size in validation['imagesize']:
    imagesize = image_size * image_size
    test.append([imagesize * 4]) #2 global mem accesses * 4 bytes


featureval = np.array(test)
print("Predictions: ")
print(model.predict(featureval))
print("Done")

y = df['h2dtime_img_transfer']
X = df[['H2dsize']] 
model = sm.OLS(y,X).fit()
summary=model.summary()
print(summary)