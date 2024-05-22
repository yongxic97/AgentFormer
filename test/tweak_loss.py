import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

total_len = 3
z0 = np.zeros(total_len)
z1 = np.zeros(total_len)

for i in range(total_len):
    num1 = np.random.rand()
    num2 = np.random.rand()
    while num1 >= num2:
        num1 = np.random.rand()
        num2 = np.random.rand()
    z0[i] = num1
    z1[i] = num2

z0 = z0[:,np.newaxis,np.newaxis]
z1 = z1[np.newaxis,:,np.newaxis]

# z0 = np.linspace(0.1, 0.9, 4)[:,np.newaxis,np.newaxis]
# z1 = np.linspace(0.1, 0.9, 4)[np.newaxis,:,np.newaxis]

# Delta_des = np.array([-10, -1, 0, 1, 10])[np.newaxis,np.newaxis,:] # des0 - des1
Delta_des = np.array([-10,10])[np.newaxis,np.newaxis,:] # des0 - des1
# since z0 < z1 is fixed, we expect to encourage Delta_des[i] > 0, and discourage Delta_des[i] < 0
# Hence, the loss should be higher for Delta_des[i] < 0, and lower for Delta_des[i] > 0

Pref = np.empty((z0.shape[0], z1.shape[1], Delta_des.shape[2]))

# check = np.tile(z1 > z0, (1, 1, Delta_des.shape[2]))

# Pref[check] = (sigmoid(Delta_des) * (z1- z0) + z0)[check]
# Pref[~check] = (sigmoid(-Delta_des) * (z0 - z1) + z1)[~check]
Pref = (sigmoid(Delta_des) * (z1 - z0) + z0)

Loss = np.log(z0) * Pref + np.log(z1) * (z0 + z1 - Pref)
Loss *= -1

good = (np.sign(z0 - z1) * np.sign(Delta_des) > 0)

print("-------- Preference --------")
print(Pref)
print("-------- Loss --------")
print(Loss)
print("-------- Good --------")
print(good)
# for datapoints with good==True, loss should be smaller compared with datapoints in the same row.