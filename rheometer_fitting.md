```py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
​
# Fractional Kelvin-Voigt model
def fractional_kelvin_voigt_model(frequency, mu_0, mu_a, alpha):
    omega = 2 * np.pi * frequency
    G_star = mu_0+mu_a*omega**alpha*np.cos(np.pi/2*alpha) + 1j*mu_a*omega**alpha*np.sin(np.pi/2*alpha)
    return G_star
​
# Define the error function
def error_function(params, frequency, storage_modulus, loss_modulus):
    mu_0, mu_a, alpha = params
    model = fractional_kelvin_voigt_model(frequency, mu_0, mu_a, alpha)
    storage_model = np.real(model)
    loss_model = np.imag(model)
    
    return np.sqrt(np.mean((storage_model - storage_modulus) ** 2 + (loss_model - loss_modulus) ** 2))
​
# Read the dataset
data = pd.read_csv("your_data_file.csv")
frequency = data.iloc[:, 0].values
storage_modulus = data.iloc[:, 1].values
loss_modulus = data.iloc[:, 2].values
​
# Initial parameter guesses
mu_00 = 1e6  # Initial guess for mu_0, adjust as needed
mu_a0 = 1e6  # Initial guess for mu_a, adjust as needed
alpha0 = 0.5  # Initial guess for alpha, adjust as needed
initial_params = [mu_00, mu_a0, alpha0]
​
# Optimize the model parameters
result = curve_fit(
    lambda frequency, mu_0, mu_a, alpha: np.hstack((np.real(fractional_kelvin_voigt_model(frequency, mu_0, mu_a, alpha)), np.imag(fractional_kelvin_voigt_model(frequency, mu_0, mu_a, alpha)))),
    frequency,
    np.hstack((storage_modulus, loss_modulus)),
    p0=initial_params,
)
​
# Extract the optimized parameters
mu_0_opt, mu_a_opt, alpha_opt = result[0]
​
print("Optimized parameters:")
print(f"mu_0: {mu_0_opt}")
print(f"mu_a: {mu_a_opt}")
print(f"alpha: {alpha_opt}")
```