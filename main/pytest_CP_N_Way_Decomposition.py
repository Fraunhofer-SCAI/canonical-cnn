## Note: In case the case of pytesting the 4-way decomposition 
## change the variables ip_shape, r and comment lines 34, 52
## and uncomment lines 35, 53


from CP_N_Way_Decomposition import CP_ALS
from tensorly.decomposition import parafac
import numpy as np
import pytest
import tensorly as tl
import time
import torch
## Defining input parameters and also creating input
ip_shape = (3, 3, 3)
r_state = 0
torch.manual_seed(r_state)
X_tensor = torch.randn(ip_shape)
max_iter = 100
r = 3
print()
print("Input Tensor: ")
print("-------------")
print(X_tensor)
## Computing the factors using our implementation of ALS
cp = CP_ALS()
start = time.time()
A, lmbds = cp.compute_ALS(X_tensor, max_iter, r)
end = time.time()
print()
print("Compüted factors")
print()

## Reconstructing the tensor from the factors
M_als_new = cp.reconstruct_Three_Way_Tensor(A[0], A[1], A[2])
#M_als_new = cp.reconstruct_Four_Way_Tensor(A[0], A[1], A[2], A[3])
print("Reconstruction tensor (implementation): ")
print("----------------------")
print(np.round(M_als_new, 3))
print()
print("Time elapsed(in seconds): ")
print("--------------------------")
print(end-start)
print()

## Computing the factors from the tensorly parafac implementation
start = time.time()
x = parafac(tl.tensor(X_tensor), rank=r, normalize_factors=False, init='random', random_state=r_state, n_iter_max= max_iter)
end = time.time()
facs = x[1]

## Reconstructing the tensor from the factors
M_fac = cp.reconstruct_Three_Way_Tensor(torch.from_numpy(facs[0]), torch.from_numpy(facs[1]), torch.from_numpy(facs[2]))
#M_fac = cp.reconstruct_Four_Way_Tensor(torch.from_numpy(facs[0]), torch.from_numpy(facs[1]), torch.from_numpy(facs[2]),  torch.from_numpy(facs[3]))
print("Reconstruction tensor (tensorly): ")
print("----------------------------------")
print(np.round(M_fac, 3))
print("Time elapsed(in seconds): ")
print("--------------------------")
print(end-start)


print()
print("Difference:")
print("-----------")
diff = np.absolute(M_fac-M_als_new).detach().numpy()
print(diff)
## PYTEST, comparing if the difference in reconstructions is same or not
## this is done by checking if the difference is < 0.01.
assert (np.all(diff<0.01))
print("**********PASS**********")
