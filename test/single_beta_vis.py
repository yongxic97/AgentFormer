import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as B

# Define the parameters alpha and beta
alpha = 9.97
beta = 29.01

# Generate a range of x values
x = np.linspace(0, 1, 100)

# Calculate the corresponding y values using the beta distribution
y = B.pdf(x, alpha, beta)

# Plot the beta distribution
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Beta Distribution (alpha={}, beta={})'.format(alpha, beta))
plt.grid(True)
plt.show()