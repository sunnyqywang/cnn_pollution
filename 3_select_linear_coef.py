import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from setup import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

output_folder = '210930'
output_var = [0]
image_radius = 30
import_hyperparameters = 153
run_suffix = "_withinit"

linear_coefs = [0,0.1,0.2,0.25,0.5,0.75,0.9,0.999]

cnn_performance_table = pd.read_csv(output_dir+output_folder+\
    '/results/cnn_performance_table_'+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+'_'+
    str(import_hyperparameters)+run_suffix+'.csv')

cnn_performance_table = cnn_performance_table[cnn_performance_table['linear_coef'].isin(linear_coefs)]
n = 15
topn = cnn_performance_table.groupby('linear_coef').head(n)
plt.scatter(topn['linear_coef'], topn['val_mse_pm25'],s=10, color='gray')
print(topn.groupby('linear_coef', as_index=False).mean()[['linear_coef','val_mse_pm25']])

poly = PolynomialFeatures(degree=3)
x_transform = poly.fit_transform(topn['linear_coef'].to_numpy()[:, np.newaxis])
reg = LinearRegression(fit_intercept=False)
reg.fit(x_transform, topn['val_mse_pm25'].to_numpy())
x = np.linspace(0,1,100)
plt.plot(x, reg.predict(poly.fit_transform(x[:, np.newaxis])), color='k')

plt.xlabel("linear coefficient")
plt.ylabel("Validation MSE")
plt.grid()
plt.xlim([0,1])

plt.savefig(output_dir+output_folder+"/lambda"+run_suffix+".png", bbox_inches='tight')
