import copy
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import time
import torch

class CP_ALS():
    """
    This class computes the Candecomp PARAFAC decomposition using 
    N-way Alternating least squares algorithm along with khatri rao product
    """
    def perform_Kronecker_Product(self, t1, t2):
        t1_flatten = torch.flatten(t1)
        op = torch.empty((0, ))
        for element in t1_flatten:
            output = element*t2
            op = torch.cat((op, output))
        return op
    
    def perform_Khatri_Rao_Product(self, t1, t2):
        # Check for criteria if the columns of both matrices are same
        r1, c1 = t1.shape
        r2, c2 = t2.shape
        if c1 != c2:
            print("Number of columns are different. Product can't be performed")
            return 0
        opt = torch.empty((r1*r2, c1))
        for col_no in range(0, t1.shape[-1]):
            x = self.perform_Kronecker_Product(t1[:, col_no], t2[:, col_no])
            opt[:, col_no] = x
        return opt
    
    def compute_MTTKRP(self, tensor_matrix, A, k_value):
        """
        This method computes the Matricized Tensor Times Khatri-Rao product
        between the unfolded tensor and the all other factors apart from kth factor.
        Input : 
            tensor_matrix : Unfolded tensor as a matrix
            A : Factor matrices
            k_value : index of kth matrix to be excluded
        Output : 
            B : Resultant MTTKRP matrix
        """
        A_matrix = copy.deepcopy(A)
        A_matrix.pop(k_value)
        krp_matrix = A_matrix[0]
        for index in range(1, len(A_matrix)):
            krp_matrix = self.perform_Khatri_Rao_Product(krp_matrix, A_matrix[index])
        B = torch.matmul(tensor_matrix, krp_matrix)
        return B
    
    def compute_V_Matrix(self, A, k_value):
        """
        This method computes the V value as a hadamard product of 
        outer product of every factort matrix apart from kth factor matrix.
        Input : 
            A : Factor matrices
            k_value : index of kth matrix to be excluded
        Output : 
            v : Resultant V matrix after the hadamard product
        """
        A_matrix = copy.deepcopy(A)
        A_matrix.pop(k_value)
        v = torch.matmul(A_matrix[0].T, A_matrix[0])
        for index in range(1, len(A_matrix)):
            p = torch.matmul(A_matrix[index].T, A_matrix[index])
            v = v*p
        return v
    
    def create_A_Matrix(self, tensor_shape, rank):
        """
        This method generates required number of factor matrices.
        Input : 
            tensor_shape : shape of the input tensor
            rank : Required rank of the factors
        Output : 
            A : Resultant list of factor matrices
        """
        A = []
        for i in tensor_shape:
            A.append(torch.randn((i, rank)))
        return A
    
    def compute_ALS(self, input_tensor, max_iter, rank):
        """
        This method is heart of this algorithm, this computes the factors and also lambdas of the algorithm.
        Input : 
            input_tensor : Tensor containing input values
            max_iter : maximum number of iterations
            rank : prescribed rank of the resultant factors
        Output : 
            A : factor matrices
            lmbds : column norms of each factor matrices
        """
        A = self.create_A_Matrix(input_tensor.shape, rank)
        lmbds = []
        for l_iter in range(0, max_iter):
            for k in range(0, len(A)):
                X_unfolded = torch.from_numpy(tl.unfold(tl.tensor(input_tensor), mode = k))
                Z = self.compute_MTTKRP(X_unfolded, A, k)
                V = self.compute_V_Matrix(A, k)
                A_k = torch.matmul(Z, torch.pinverse(V))
                l = torch.norm(A_k, dim=0)
                d_l = np.zeros((rank, rank))
                np.fill_diagonal(d_l, l)
                #A_k = np.dot(A_k, np.linalg.pinv(d_l))
                if l_iter == 0:
                    lmbds.append(np.linalg.norm(l))
                else:
                    lmbds[k] = np.linalg.norm(l)
                A[k] = A_k
        return A, lmbds
    
    def reconstruct_tensor(self, factors, norm, rank, ip_shape):
        """
        This method reconstructs the tensor given factor matrices and norms
        Input : 
            factors : factor matrices
            norm : column norms of every factor matrices
            rank : prescribed rank of the resultant factors
            ip_shape : Input tensor shape 
        Output : 
            M : Reconstructed tensor
        """
        M = 0       
        for c in range(0, rank):
            op = factors[0][:, c]
            for i in range(1, len(factors)):
                op = np.outer(op.T, factors[i][:, c])
            M += op
        M = np.reshape(M, ip_shape)
        return M

    def reconstruct_Three_Way_Tensor(self, a, b, c):
        """This method reconstructs the tensor from the rank one factor matrices
        Inputs: 
            a : First factor in CP decomposition
            b : Second factor in CP decomposition
            c : Third factor in CP decomposition
        Output:
            x_t : Reconstructed output tensor"""

        x_t = 0
        #row, col = a.shape()
        for index in range(a.shape[1]):
            x_t += torch.ger(a[:,index], b[:,index]).unsqueeze(2)*c[:,index].unsqueeze(0).unsqueeze(0)
        return x_t

    # Reconstruct the tensor from the factors
    def reconstruct_Four_Way_Tensor(self, a, b, c, d):
        """This method reconstructs the tensor from the rank one factor matrices
        Inputs: 
            a : First factor in CP decomposition
            b : Second factor in CP decomposition
            c : Third factor in CP decomposition
            d : Fourth factor in CP decomposition
        Output:
            x_t : Reconstructed output tensor"""

        x_t = 0
        #row, col = a.shape()
        for index in range(a.shape[1]):
            Y = (torch.ger(a[:, index], b[:, index]).unsqueeze(2)*c[:, index]).unsqueeze(3)*d[:,index].unsqueeze(0).unsqueeze(0)
            x_t += Y
            #x_t += torch.ger(a[:,index], b[:,index]).unsqueeze(2)*c[:,index].unsqueeze(0).unsqueeze(0)
        return x_t
    
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

cp = CP_ALS()
start = time.time()
A, lmbds = cp.compute_ALS(X_tensor, max_iter, r)
end = time.time()
print()
print("Compüted factors")
#M_als = cp.reconstruct_tensor(A, lmbds, r, ip_shape)
#print("Reconstruction tensor: ")
#print("----------------------")
#print(np.round(M_als, 3))
print()

M_als_new = cp.reconstruct_Three_Way_Tensor(A[0], A[1], A[2])
#M_als_new = cp.reconstruct_Four_Way_Tensor(A[0], A[1], A[2], A[3])
print("Reconstruction tensor (implementation): ")
print("----------------------")
print(np.round(M_als_new, 3))
print()
print("Time elapsed(in seconds): ")
print("--------------------------")
print(end-start)

start = time.time()
x = parafac(tl.tensor(X_tensor), rank=r, normalize_factors=False, init='random', random_state=r_state, n_iter_max= max_iter)
end = time.time()
facs = x[1]
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
#print(diff)
print(np.any(diff>0.01))
if np.any(diff>0.01):
    print("Fail")
else:
    print("Pass")
    
#print()
#print()
#print("Reconstruction from the library")
#print(np.round(tl.kruskal_to_tensor(x), 3))