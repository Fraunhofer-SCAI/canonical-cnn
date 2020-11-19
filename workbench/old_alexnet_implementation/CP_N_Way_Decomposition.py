import copy
import numpy as np
import tensorly as tl
tl.set_backend("pytorch")
import torch 
import matplotlib 
import time
matplotlib.use('Agg')
import pylab as plt
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
    
    #def perform_Kronecker_Product(self, t1, t2):
    #    t1_flatten = torch.flatten(t1)
    #    op = torch.empty((0, ))
    #    for element in t1_flatten:
    #        output = element*t2
    #        op = torch.cat((op, output))
    #    return op
    
    #def perform_Khatri_Rao_Product(self, t1, t2):
    #    # Check for criteria if the columns of both matrices are same
    #    r1, c1 = t1.shape
    #    r2, c2 = t2.shape
    #    if c1 != c2:
    #        print("Number of columns are different. Product can't be performed")
    #        return 0
    #    opt = torch.empty((r1*r2, c1))
    #    for col_no in range(0, t1.shape[-1]):
    #        x = self.perform_Kronecker_Product(t1[:, col_no], t2[:, col_no])
    #        opt[:, col_no] = x
    #    return opt

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
        #A_matrix = copy.deepcopy(A)
        #A_matrix.pop(k_value)
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
        #A_matrix = copy.deepcopy(A)
        #A_matrix.pop(k_value)
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
    def compute_fit(self, X, M):
        diff = X-M
        fit = (np.linalg.norm(diff))/(np.linalg.norm(X))
        return fit
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
        fit_list = []
        for l_iter in range(0, max_iter):
            for k in range(0, len(A)):
                X_unfolded = self.unfold_tensor(input_tensor, k)
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
            if l_iter%5 == 0:
                M = self.reconst(A, input_tensor.shape)
                fit_val = self.compute_fit(input_tensor, M)
                fit_list.append([l_iter, fit_val])
        return A, lmbds, fit_list
    
    def reconst(self, A, ip_shape):
        tensor = torch.matmul(A[0], tl.tenalg.khatri_rao(A, skip_matrix=0).T)
        return tl.fold(tensor, 0, ip_shape)
    
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


threshold = 2e-7
als = CP_ALS()
x = torch.randn((2, 3, 3))
weight_tensor = x
print(weight_tensor.shape, " SHAPEmdkahbfjdhsafvjsad")
start = time.time()
A, lmbds, fits = als.compute_ALS(weight_tensor, 200, 2)
end = time.time()
print(fits, flush=True)
print()
print("Time taken to execute is: ",abs(start-end), flush=True)
print(flush=True)
poss_ranks = [64, 192, 256, 320, 384, 448, 512]
colors = ['r', 'b', 'k', 'g', 'c', 'm', 'y']
poss_ranks = [1, 2, 3, 4, 5]
poss_fits = []
for rank in poss_ranks:
    print("Computing rank: ", rank, flush=True)
    print()
    _, _, fit = als.compute_ALS(weight_tensor, 200, rank)
    poss_fits.append(fits)

print("Length: ", len(poss_fits), flush=True)
plt.figure(figsize=(10, 10))
precent = []
for index, f_list in enumerate(poss_fits):
    x = []
    y = []
    clr = colors[index]
    for i in f_list:
        plt.scatter(i[0], i[1], c=clr)
        x.append(i[0])
        y.append(i[1])
    lbl = "rank : "+str(poss_ranks[index])
    plt.plot(x, y, c=clr, label = lbl)
    precent.append(f_list[-1][-1])
plt.legend()
plt.xlabel("Number of iterations")
plt.ylabel("Fit")
plt.grid()
plt.savefig("Multiple_ranks.png")

plt.figure(figsize=(10, 10))
plt.scatter(poss_ranks, precent, c='r', s=80)
plt.plot(poss_ranks, precent)
plt.xlabel("Rank")
plt.ylabel("Fit")
plt.grid()
plt.savefig("RankVSPercent.png")