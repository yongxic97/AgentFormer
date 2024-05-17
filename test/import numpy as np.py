import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

z0 = np.linspace(0.1, 0.9, 4)[:,np.newaxis,np.newaxis]
z1 = np.linspace(0.1, 0.9, 4)[np.newaxis,:,np.newaxis]

Delta_des = np.array([-10, -1, 0, 1, 10])[np.newaxis,np.newaxis,:] # des0 - des1

Pref = np.empty((z0.shape[0], z1.shape[1], Delta_des.shape[2]))

check = np.tile(z1 > z0, (1, 1, Delta_des.shape[2]))

Pref[check] = (sigmoid(Delta_des) * (z1- z0) + z0)[check]
Pref[~check] = (sigmoid(-Delta_des) * (z0 - z1) + z1)[~check]

Loss = np.log(z1) * Pref + np.log(z0) * (z0 + z1 - Pref)
Loss *= -1

good = (np.sign(z0 - z1) * np.sign(Delta_des) > 0)

print(Loss)
print(good)