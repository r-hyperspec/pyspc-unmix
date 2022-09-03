# `pyspc-unmix`: Python package for unmixing hyperspectral data

<!-- badges: start -->
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
<!-- badges: end -->

# Installation

```bash
pip install git+https://github.com/r-hyperspec/pyspc-unmix
```

# Available alogrithms
* N-FINDR

# Example

Generate pure signals
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pyspc_unmix import NFINDR

# Prepare pure signals
x = np.arange(0, 20, 0.01)
y1 = stats.norm.pdf(x, 7, 1)
y2 = stats.norm.pdf(x, 10, 1)
y3 = stats.norm.pdf(x, 13, 1)

# Plot the signals
fig, ax = plt.subplots()
ax.plot(x, y1, "r-.")
ax.plot(x, y2, "g-.")
ax.plot(x, y3, "b-.")
plt.show()
```
![Figure_1](https://user-images.githubusercontent.com/9852534/188256343-e4a642fb-e670-4433-b5f1-0d4f584f9023.png)



Mix the pure signals
```python
y1_coefs = [0, 0.5, 1, 1.5, 2]
y2_coefs = [0, 0.5, 1, 1.5, 2]
y3_coefs = [0, 0.5, 1, 1.5, 2]

# Make all possible combination of the coefficients
true_coefs = np.array(np.meshgrid(y1_coefs, y2_coefs, y3_coefs)).T.reshape(-1, 3)
# Replace (0,0,0) point to avoid zero division
true_coefs[0, :] = [0.1, 0.1, 0.1]
# Normalize coefficients so the sum is always 1
true_coefs = (true_coefs.T / np.sum(true_coefs.T, 0)).T
np.random.shuffle(true_coefs)

print(np.round(true_coefs[:5,:],2))
# array([[0.5 , 0.  , 0.5 ],
#        [0.57, 0.  , 0.43],
#        [0.67, 0.33, 0.  ],
#        [0.2 , 0.6 , 0.2 ],
#        [0.4 , 0.  , 0.6 ]])


# Plot the mixtures
mixtures = true_coefs @ np.vstack((y1, y2, y3))
fig, ax = plt.subplots()
for i in range(len(mixtures)):
    ax.plot(x, mixtures[i, :])
plt.show()
```
![Figure_2](https://user-images.githubusercontent.com/9852534/188256425-86057096-ae6e-41fb-99b8-c36ba136f637.png)


Apply NFINDR

```python
# First, reduce dimension with PCA
x = PCA(n_components=2).fit_transform(mixtures)

# Apply NFINDR to find pure components and the concentrations
# Set random state for reproducible results
nf = NFINDR(n_endmembers=3, random_state=21)
nf.fit(x)

# See found endmembers
print(true_coefs[nf.endmember_indecies_,:])
# array([[0., 0., 1.], 
#       [1., 0., 0.], 
#       [0., 1., 0.]])

print(np.round(nf.endmembers_,2))
# [[-3.76 -2.01]
#  [ 3.76 -2.01]
#  [-0.    4.02]]

# Calculate the coefficients/concetrations
nfindr_coefs = nf.transform(x)

# The values correspond to true concentrations
print(np.round(nfindr_coefs[:5,:],2))
# array([[0.5 , 0.5 , 0.  ],
#        [0.43, 0.57, 0.  ],
#        [0.  , 0.67, 0.33],
#        [0.2 , 0.2 , 0.6 ],
#        [0.6 , 0.4 , 0.  ]])

```

## Pipline interface
```python
from sklearn.pipeline import Pipeline

p=3
pca_nfindr_pipe = Pipeline([('pca', PCA(n_components=p-1)), ('nfindr', NFINDR(n_endmembers=p, random_state=21))])
nfindr_coefs = pca_nfindr_pipe.fit_transform(mixtures)

print(np.round(nfindr_coefs[:5,:],2))
# array([[0.5 , 0.5 , 0.  ],
#        [0.43, 0.57, 0.  ],
#        [0.  , 0.67, 0.33],
#        [0.2 , 0.2 , 0.6 ],
#        [0.6 , 0.4 , 0.  ]])
```
