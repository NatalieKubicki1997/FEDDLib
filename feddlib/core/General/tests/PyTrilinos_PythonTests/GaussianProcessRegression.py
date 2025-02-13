import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,  WhiteKernel

# Fitted Function from Group of Professor Gommper
def viscosity_blood(gamma, lambda_, n, mu_0, mu_inf):
    return mu_inf + (mu_0 - mu_inf) * (1 + (gamma * lambda_)**2)**((n-1)/2)

# decide if data set for blood without plasma or with plasma
with_plasma = 0

if with_plasma == 0:
    viscosity_blood_without_plasma = [3.38572530000000, 3.47924380000000, 3.62368100000000, 3.79070950000000, 4.20250300000000, 5.25822800000000,
                                      6.01490200000000, 6.81933900000000, 7.97732970000000, 8.88347300000000, 9.58859350000000, 9.85210300000000,
                                      10.7784050000000, 11.3762850000000, 11.9546880000000, 12.3391410000000, 12.9138090000000, 13.3343040000000]
    shear_rate_without_plasma = [1311.94130000000, 665.174560000000, 324.603850000000, 179.388240000000, 102.040950000000, 33.6564800000000,
                                 17.9062080000000, 9.34580800000000, 4.56230700000000, 2.62014820000000, 1.35431800000000, 0.847528800000000,
                                 0.390575620000000, 0.220022540000000, 0.110499950000000, 0.0616510180000000, 0.0167842490000000, 0.00512516360000000]

    shear_rate_data_o = np.flip(shear_rate_without_plasma)
    shear_rate_data = shear_rate_data_o
    viscosity_data_o = np.flip(viscosity_blood_without_plasma)
    viscosity_data = viscosity_data_o

    # Blood without Plasma - Parameters
    lambda_ = -1
    n = 0.5
    mu_0 = 13.3343
    mu_inf = 3.3857

    # Very fine discretized to ensure high resolution
    gamma = np.arange(shear_rate_data[0], shear_rate_data[-1]+0.005, 0.005)
    viscosity_function_evaluated = viscosity_blood(gamma, lambda_, n, mu_0, mu_inf)


log_shear_rate_data = np.log(shear_rate_data)
log_viscosity_data = np.log(viscosity_data)
log_viscosity_function = np.log(viscosity_function_evaluated)
log_gamma = np.log(gamma)


# Define kernel function
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e5))

# Train the Gaussian process regression model
log_viscosity_GP = GaussianProcessRegressor(kernel=kernel).fit(log_shear_rate_data.reshape(-1,1), log_viscosity_data)


# Predict
log_viscosity_function_predict, xtest_std = log_viscosity_GP.predict(log_gamma.reshape(-1,1), return_std=True)
xtestci = np.column_stack((log_viscosity_function_predict - 1.96*xtest_std, log_viscosity_function_predict + 1.96*xtest_std))



plt.plot(log_shear_rate_data, log_viscosity_data, '*', linewidth=1.5, color='green')
plt.plot(log_gamma, log_viscosity_function, linewidth=2.5, color='red')

# Predicted Trajectory vs real
plt.plot(log_gamma, log_viscosity_function_predict, linewidth=1.5, color='blue')
plt.plot(log_gamma, xtestci[:,0], 'k:')
plt.plot(log_gamma, xtestci[:,1], 'k:')
plt.legend(['Log of Simulation data points', 'Log of Carreau Fitted Curve', 'GPR predictions', '95% lower', '95% upper'], loc='best')

plt.figure()
plt.plot(shear_rate_data, viscosity_data, '*', color='green', linewidth=1.5)
plt.plot(gamma, viscosity_function_evaluated, color='red', linewidth=2.5)
plt.plot(gamma, np.exp(log_viscosity_function_predict), linewidth=1.5, color='blue')
plt.plot(gamma, np.exp(xtestci[:,0]), 'k:')
plt.plot(gamma, np.exp(xtestci[:,1]), 'k:')
plt.legend(['Simulation data points original', 'Carreau Fitted Curve original', 'Exp of GPR predictions', '95% lower', '95% upper'], loc='best')

plt.show()

plt.figure()
plt.loglog(shear_rate_data, viscosity_data, '*', color='green', linewidth=1.5)
plt.loglog(gamma, viscosity_function_evaluated, color='red', linewidth=2.5)
plt.loglog(gamma, np.exp(log_viscosity_function_predict), linewidth=1.5, color='blue')
plt.loglog(gamma, np.exp(xtestci[:,0]), 'k:')
plt.loglog(gamma, np.exp(xtestci[:,1]), 'k:')
plt.legend(['Simulation data points (loglog scale plotted)', 'Carreau Fitted Curve original', 'Exp of GPR predictions', '95% lower', '95% upper'], loc='best')