# Libraries needed
import pytest
import tensorly
import torch
from tensorly.decomposition import parafac

# Reconstruct the tensor from the factors
def reconstruct_Three_Way_Tensor(a, b, c):
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

# Compute the 3-way CP decomposition using Adam optimizer
def three_Way_CP_Decomposition(X, rank, max_iter, l_rate, random_state):
    """This method calculates the required gradient to change the factor matrices.
    This gradient is calculated based on the Forbenius norm between the original tensor and reconstructed tensor.
    An ADAM optimizer is used for the caluculation of gradients.
    Inputs: 
        X : Input tensor
        rank : Rank of the tensor
        max_iter : Maximum iterations for the optimizer to run
        l_rate : Learning rate for the optimizer
        random_state : A variable to stop the randomness. If set to integer, randomness becomes deterministic.
    Outputs:
        losses : A list containing the loss calculated during each epoch
        factors : A list containing the factor matrices a, b and c."""
    
    torch.manual_seed(random_state)
    a = torch.randn((X.shape[0],rank), requires_grad=True)
    b = torch.randn((X.shape[1],rank), requires_grad=True)
    c = torch.randn((X.shape[2],rank), requires_grad=True)
    factors = [a, b, c]
    ADAM_optimizer = torch.optim.Adam(factors, lr = l_rate)
    losses = []
    for index in range(max_iter):
        x_t = reconstruct_Three_Way_Tensor(*factors)
        ADAM_optimizer.zero_grad()
        loss = torch.mean((X-x_t)**2)
        losses.append(loss.item())
        loss.backward(retain_graph=True)
        ADAM_optimizer.step()
    
    return losses, factors

# Test the output with the parafac decomposition from the tensorly library
def test_outputs(input_tensor_shape, r, max_iterations, l_rate, random_state = 0):
    """This method unit test the implemented 3-way decomposition with the 
    standard implementation from tensorly library.
    Inputs: 
        input_tensor_shape : A tuple stating the input shape of the tensor.
        r :  Specified rank of the tensor.
        max_iterations : Maximum iterations for the optimizer to run
        l_rate : Learning rate for the optimizer
        random_state : A variable to stop the randomness. If set to integer, randomness becomes deterministic.
    Output:
    """
    print("In test output")
    torch.manual_seed(random_state)
    input_tensor = torch.randn(input_tensor_shape)
    w, factors = parafac(tensorly.tensor(input_tensor), r, max_iterations)
    print("parafac dome")
    outputs = three_Way_CP_Decomposition(input_tensor, r, max_iterations, l_rate, random_state)
    print("3 way done")
    loss = outputs[0]
    facs = outputs[1]
    print("##########")
    print("Factors from the implementation: ")
    print("a:")
    print(facs[0])
    print("b:")
    print(facs[1])
    print("c:")
    print(facs[2])
    print("##########")
    print("Factors from the tensorly: ")
    print("a:")
    print(factors[0])
    print("b:")
    print(factors[1])
    print("c:")
    print(factors[2])
    print("##########")
    print("PYTEST RESULTS: ")
    try:
        assert facs[0].numpy() == factors[0]
        print("First factor matched")
    except:
        print("First factor didn't match")
    try:
        assert facs[1].numpy() == factors[1]
        print("Second factor matched")
    except:
        print("Second factor didn't match")
    try:
        assert facs[2].numpy() == factors[2]
        print("Third factor matched")
    except:
        print("Third factor didn't match")


input_shape = (3, 6, 5)
rank = 2
max_iter = 100
learning_rate = 0.1
test_outputs(input_shape, rank, max_iter, learning_rate)