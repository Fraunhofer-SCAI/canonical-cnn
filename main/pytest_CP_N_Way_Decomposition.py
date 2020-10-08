## Note: In case the case of pytesting the 4-way decomposition 
## change the variables ip_shape, r and comment lines 34, 52
## and uncomment lines 35, 53


from CP_N_Way_Decomposition import CP_ALS
from tensorly.decomposition import parafac
import numpy as np
import pytest
import tensorly as tl
tl.set_backend('pytorch')
import time
import torch

class pytest_ALS():

    def __init__(self, ip_shape, random_state, max_iter, rank):
        super().__init__()
        torch.manual_seed(random_state)
        self.X_tensor = torch.randn(ip_shape)
        self.max_iter = max_iter
        self.rank = rank
        self.random_state = random_state
        self.cp_als = CP_ALS()
    
    def pytest_Unfolding(self):
        shapes = [(3, 3, 2), (3, 4, 3, 2), (3, 4, 3, 3, 2)]
        for shape in shapes:
            print()
            print("Testing for shape: ", shape)
            print()
            Tensor = torch.randn(shape)
            range_val = len(shape)
            for col_no in range(0, range_val):
                print()
                print("\t Mode: ", col_no)
                print()
                u1 = tl.unfold(Tensor, col_no)
                u2 = self.cp_als.unfold_tensor(Tensor, col_no)
                #assert u1.all() == u2.all()
                assert torch.all(torch.eq(u1, u2))
                print()
                print("******* UNFOLDING TESTCASE PASSED *******")
                print()

    def pytest_Reconstruction(self, init_type):
        A, lmbds = self.cp_als.compute_ALS(self.X_tensor, self.max_iter, self.rank)
        M_als_new = self.cp_als.reconstruct_Three_Way_Tensor(A[0], A[1], A[2])
        x = parafac(self.X_tensor, rank=self.rank, normalize_factors=False, init=init_type, random_state=self.random_state, n_iter_max= self.max_iter)
        facs = x[1]
        M_fac = self.cp_als.reconstruct_Three_Way_Tensor(facs[0], facs[1], facs[2])
        print()
        print("Difference:")
        print("-----------")
        diff = np.absolute(M_fac-M_als_new).detach().numpy()
        print(diff)
        assert (np.all(diff<0.01))
        print()
        print("********** RECONSTRUCTION TESTCASE PASSED **********")

ip_shape = (3, 3, 3)
r_state = 0
max_iter = 1000     
r = 3
ptest_ALS = pytest_ALS(ip_shape, r_state, max_iter, r)
print()
print("TESTCASE 1 : Testing the unfolding of the matrix......")
ptest_ALS.pytest_Unfolding()
print()
print("TESTCASE 2 : Testing the reconstruction of tensor from factors.......")
ptest_ALS.pytest_Reconstruction('random')
print()