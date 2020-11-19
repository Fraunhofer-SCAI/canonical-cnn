import copy
from os import device_encoding
import numpy as np
import tensorly as tl
tl.set_backend("pytorch")
import torch 
import time
from alexnet_model import AlexNet
import matplotlib 
matplotlib.use('Agg')
import pylab as plt
from tensorly.decomposition import parafac

class CP_ALS():
    """
    This class computes the Candecomp PARAFAC decomposition using 
    N-way Alternating least squares algorithm along with khatri rao product
    """
    def moveaxis(self, tensor, source, destination):
        """
        This method is from the implementation given in pytorch https://github.com/pytorch/pytorch/issues/36048#issuecomment-652786245
        Input : 
            tensor : Input tensor
            source : First axis to move
            destination : Second axis to replace the first one
        Output :
            Output tensor to where the axis is moved
        """
        dim = tensor.dim()
        perm = list(range(dim))
        if destination < 0:
            destination += dim
        perm.pop(source)
        perm.insert(destination, source)
        return tensor.permute(*perm)
    
    def unfold_tensor(self, tensor, mode):
        """ This method unfolds the given input tensor along with the specified mode.
        Input :
            tensor : Input tensor
            mode : Specified mode of unfolding
        Output :
            matrix : Unfolded matrix of the tensor with specified mode
        """
        t = self.moveaxis(tensor, mode, 0)
        matrix = t.reshape(tensor.shape[mode], -1)
        return matrix 

    def compute_fit(self, X, M):
        diff = X-M
        fit = (torch.norm(diff))/(torch.norm(X))
        return fit

    def perform_Kronecker_Product(self, A, B):
        """ 
        This method performs the kronecker product of the two matrices
        The method is adaption of the method proposed in https://discuss.pytorch.org/t/kronecker-product/3919/10
        Input : 
            A : Input matrix 1
            B : Input matrix 2
        Output : 
            Output is the resultant matrix after kronecker product
        """
        return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))
    
    def perform_Khatri_Rao_Product(self, A, B):
        """
        This methods performs the Khatri Rao product as it is the column wise kronecker product
        Input : 
            A : Input matrix 1
            B : Input matrix 2
        Output : 
            result : The resultant Khatri-Rao product matrix
        """
        if A.shape[1] != B.shape[1]:
            print("Inputs must have same number of columns")
            return 0
        result = None
        for col in range(A.shape[1]):
            res = self.perform_Kronecker_Product(A[:, col].unsqueeze(0), B[:, col].unsqueeze(0))
            if col == 0:
                result = res
            else:
                result = torch.cat((result, res), dim = 0)
        return result.T

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
        krp_matrix = A[0]
        for index in range(1, len(A)):
            krp_matrix = self.perform_Khatri_Rao_Product(krp_matrix, A[index])
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
        v = torch.matmul(A[0].T, A[0])
        for index in range(1, len(A)):
            p = torch.matmul(A[index].T, A[index])
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
    
    def compute_ALS(self, input_tensor, max_iter, rank, device, threshold):
        """
        This method is heart of this algorithm, this computes the factors and also lambdas of the algorithm.
        Input : 
            input_tensor : Tensor containing input values
            max_iter : maximum number of iterations
            rank : prescribed rank of the resultant factors
            device : Prescribed device "GPU/CPU"
        Output : 
            A : factor matrices
            lmbds : column norms of each factor matrices
        """
        A = self.create_A_Matrix(input_tensor.shape, rank)
        if device != "cpu":
            A = [tensor.to(device) for tensor in A]
        lmbds = []
        fit_list = []
        fit_val = 0.0
        for l_iter in range(0, max_iter):
            old_fit = fit_val
            for k in range(0, len(A)):
                X_unfolded = self.unfold_tensor(input_tensor.to(device), k)
                A.pop(k)
                Z = self.compute_MTTKRP(X_unfolded, A, k)
                V = self.compute_V_Matrix(A, k)
                A_k = torch.matmul(Z, torch.pinverse(V))
                #l = torch.norm(A_k, dim=0)
                #d_l = np.zeros((rank, rank))
                #np.fill_diagonal(d_l, l)
                #A_k = np.dot(A_k, np.linalg.pinv(d_l))
                #if l_iter == 0:
                #    lmbds.append(np.linalg.norm(l))
                #else:
                #    lmbds[k] = np.linalg.norm(l)
                #A[k] = A_k
                
                A.insert(k, A_k)
            M = self.reconst(A, input_tensor.shape)
            fit_val = 1-self.compute_fit(input_tensor.to(device), M.to(device))
            if l_iter%5 == 0:
                fit_list.append([l_iter, fit_val])
            if abs(old_fit-fit_val) < threshold:
                print("Fit didn't improve", flush=True)
                break
        return A, lmbds, fit_list
    
    def reconst(self, A, ip_shape):
        tensor = torch.matmul(A[0], tl.tenalg.khatri_rao(A, skip_matrix=0).T)
        return tl.fold(tensor, 0, ip_shape)

device = "cpu"
#if torch.cuda.is_available():  
  #device = "cuda:0" 
print("Working on device: ", device, flush=True)
model = AlexNet()
weight_tensor = model.features[3].weight.data
threshold = 2e-7
als = CP_ALS()
x = torch.randn((2, 3, 3))
#weight_tensor = x
print(weight_tensor.shape, " SHAPEmdkahbfjdhsafvjsad")
start = time.time()
#A, lmbds, fits = als.compute_ALS(weight_tensor, 200, 2, device, threshold)
end = time.time()
#print(fits, flush=True)
print()
print("Time taken to execute is: ",abs(start-end), flush=True)
print(flush=True)
poss_ranks = [64, 192, 256, 320, 384, 448, 512]
colors = ['r', 'b', 'k', 'g', 'c', 'm', 'y']
#poss_ranks = [1, 2, 3, 4, 5]
poss_fits = []
for rank in poss_ranks:
    print("Computing rank: ", rank, flush=True)
    print()
    #_, _, fit = als.compute_ALS(weight_tensor, 200, rank, device, threshold)
    A = parafac(weight_tensor, rank, n_iter_max = 200, init="random", normalize_factors=False)[1]
    M = als.reconst(A, weight_tensor.shape)
    fit = als.compute_fit(M, weight_tensor)
    poss_fits.append(float(1-fit))

print("Length: ", len(poss_fits), flush=True)
#plt.figure(figsize=(10, 10))
#precent = []
#for index, f_list in enumerate(poss_fits):
#    x = []
#    y = []
#    clr = colors[index]
#    for i in f_list:
#        plt.scatter(i[0], i[1].cpu(), c=clr)
#        x.append(i[0])
#        y.append(i[1].cpu())
#    lbl = "rank : "+str(poss_ranks[index])
#    plt.plot(x, y, c=clr, label = lbl)
#    precent.append(f_list[-1][-1].cpu().numpy())
#plt.legend()
#plt.xlabel("Number of iterations")
#plt.ylabel("Fit")
#plt.grid()
#plt.savefig("Multiple_ranks.png")
print(poss_fits)
plt.figure(figsize=(10, 10))
plt.scatter(poss_ranks, poss_fits, c='r', s=80)
plt.plot(poss_ranks, poss_fits)
plt.xlabel("Rank")
plt.ylabel("Fit")
plt.grid()
plt.savefig("RankVSPercent.png")