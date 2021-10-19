import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from setup import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

plt.rcParams.update({'font.size':16})

output_folder = '211010'
output_var = [0]
image_radius = 30
import_hyperparameters = 145
run_suffix = "_linearfix"

linear_coefs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,0.95]

cnn_performance_table = pd.read_csv(output_dir+output_folder+\
    '/results/cnn_performance_table_'+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+'_'+
    str(import_hyperparameters)+run_suffix+'.csv')

cnn_performance_table = cnn_performance_table[cnn_performance_table['linear_coef'].isin(linear_coefs)]
print(cnn_performance_table.groupby('linear_coef', as_index=False).count())

n = 10
topn = cnn_performance_table.groupby('linear_coef').head(n)
plt.scatter(topn['linear_coef'], topn['val_mse_pm25'], s=10, color='#6CA30F', alpha=0.6)
plt.scatter(topn['linear_coef'], topn['train_mse_pm25'], s=10, color='#0E72CC', alpha=0.6)
plt.scatter(topn['linear_coef'], topn['test_mse_pm25'], s=10, color='#F59311', alpha=0.6)
print(topn.groupby('linear_coef', as_index=False).mean()[['linear_coef','val_mse_pm25']])

linear_coefs = np.array(linear_coefs)
poly = PolynomialFeatures(degree=2)
x_transform = poly.fit_transform(topn['linear_coef'].to_numpy()[:, np.newaxis])
reg = LinearRegression(fit_intercept=False)

# errorbar_min = topn.groupby('linear_coef').min()['val_mse_pm25'].to_numpy()
# errorbar_max = topn.groupby('linear_coef').max()['val_mse_pm25'].to_numpy()
# errorbar = np.vstack((errorbar_min, errorbar_max))
reg.fit(x_transform, topn['val_mse_pm25'].to_numpy())
x = np.linspace(0,1,100)
plt.plot(x, reg.predict(poly.fit_transform(x[:, np.newaxis])), color='#6CA30F', label='Validation')
# pred = reg.predict(poly.fit_transform(linear_coefs[:, np.newaxis]))
# errorbar = np.abs(errorbar - pred)
# plt.errorbar(linear_coefs, pred, yerr=errorbar, alpha=0.6, color='k', fmt='none')
#
# errorbar_min = topn.groupby('linear_coef').min()['train_mse_pm25'].to_numpy()
# errorbar_max = topn.groupby('linear_coef').max()['train_mse_pm25'].to_numpy()
# errorbar = np.vstack((errorbar_min, errorbar_max))
reg.fit(x_transform, topn['train_mse_pm25'].to_numpy())
x = np.linspace(0,1,100)
plt.plot(x, reg.predict(poly.fit_transform(x[:, np.newaxis])), color='#0E72CC', label='Train')
# pred = reg.predict(poly.fit_transform(linear_coefs[:, np.newaxis]))
# errorbar = np.abs(errorbar - pred)
# plt.errorbar(linear_coefs, pred, yerr=errorbar, alpha=0.6, color='cornflowerblue', fmt='none')
#
# errorbar_min = topn.groupby('linear_coef').min()['test_mse_pm25'].to_numpy()
# errorbar_max = topn.groupby('linear_coef').max()['test_mse_pm25'].to_numpy()
# errorbar = np.vstack((errorbar_min, errorbar_max))
reg.fit(x_transform, topn['test_mse_pm25'].to_numpy())
x = np.linspace(0,1,100)
plt.plot(x, reg.predict(poly.fit_transform(x[:, np.newaxis])), color='#F59311', label='Test')
# pred = reg.predict(poly.fit_transform(linear_coefs[:, np.newaxis]))
# errorbar = np.abs(errorbar - pred)
# plt.errorbar(linear_coefs, pred, yerr=errorbar, alpha=0.6, color='coral', fmt='none')

plt.xlabel("Linear Coefficient")
plt.ylabel("MSE")
plt.grid()
plt.xlim([-0.05,1])
plt.legend()
plt.savefig(output_dir+output_folder+"/lambda"+run_suffix+".png", bbox_inches='tight')
