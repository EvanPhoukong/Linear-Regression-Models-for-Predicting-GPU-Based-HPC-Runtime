import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv('PA4training(1).csv')
validation = pd.read_csv('PA4validation.csv')

conv_bytes = []
cov_bytes = []
conv_flops = []
cov_flops = []

for image_size in df['imagesize']:
    imagesize = image_size * image_size
    conv_flops.append(imagesize * (3 * 2)) #Kernel size * FLOPS
    conv_bytes.append(imagesize * (8)) #2 global mem accesses * 4 bytes
    cov_flops.append(imagesize * (49 * 6 + 7)) # Window size * flops in window + eigen calcs
    cov_bytes.append(imagesize * (8)) #2 global mem accesses * 4 bytes

df['covariance_bytes'] = cov_bytes
df['convolution_bytes'] = conv_bytes
df['covariance_flops'] = cov_flops
df['convolution_flops'] = conv_flops

y = df['cov_timex1']
X = df[['covariance_flops', 'covariance_bytes']] 

print(y)
print(X) 

plt.figure(figsize=(10,6))
plt.scatter(df['covariance_flops'],y,color='blue',label='Data points')
plt.title('Covariance Time vs. FLOPS')
plt.xlabel('flops')
plt.ylabel('cov_time')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(df['covariance_bytes'],y,color='blue',label='Data points')
plt.title('Covariance Time vs. Bytes')
plt.xlabel('bytes')
plt.ylabel('cov_time')
plt.legend()
plt.grid(True)
plt.show()


model=LinearRegression()
model.fit(X,y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

test = []

for image_size in validation['imagesize']:
    imagesize = image_size * image_size
    test.append([imagesize * (49 * 6 + 7), imagesize * (8)]) #2 global mem accesses * 4 bytes


featureval = np.array(test)
print("Predictions: ")
print(model.predict(featureval))
print("Done")

y = df['cov_timex1']
X = df[['covariance_flops', 'covariance_bytes']] 
model = sm.OLS(y,X).fit()
summary=model.summary()
print(summary)
