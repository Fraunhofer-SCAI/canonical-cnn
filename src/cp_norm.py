# =====================================================================
# CP Norm application
# =====================================================================

from importlib.metadata import requires
from typing import Any, TypeVar

import tensorly as tl
tl.set_backend('pytorch')
from tensorly.decomposition import parafac, CPPower
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np



def estimate_rank(tensor: torch.Tensor, max_it: int = 1000 ) -> int:
    """This function estimates the rank of the tensor

    Args:
        tensor (torch.Tensor): Input tensor to estimate the rank
        max_it (int, optional): Upper bound for rank. Defaults to 1000.

    Returns:
        int: Best rank
    """
    for it in range(1, max_it, 2):
        try:
            decomposition = parafac(tensor, rank=it, init='random',
                                    random_state = 0)
            reconstruction = tl.cp_to_tensor(decomposition)
            err = torch.mean(torch.abs(reconstruction - tensor))
            print(it, err.item(), flush=True)
            # if err < torch.finfo(torch.float32).eps:
            if err < 1e-5:
                break
        except:
            print('rank', it, 'failed to converge.')
    return it


class CPNorm(object):
    """
    CP Norm declaration and its forward pass application

    Args:
        name [str]: Name of the module to apply CP Norm

    Raises:
        RuntimeError: If CPNorm is applied then error is raised

    Returns:
        [module]: CPNorm applied module
    """
    name : str

    def __init__(self, name : str) -> None:
        self.name = name

    
    def fill_multivariate_normal(self, matrix_):
        value = matrix_.shape[0]*matrix_.shape[1]
        n1 = value//2
        n2 = value-n1
        x1 = torch.empty(n1)
        x2 = torch.empty(n2)
        x1 = torch.nn.init.trunc_normal_(x1, 0, 0.25)
        x1 = 1-abs(x1)
        x2 = torch.nn.init.trunc_normal_(x2, -0.125, 0.3)
        x = torch.cat([x1,x2])
        perm = torch.randperm(x.shape[0])
        x = x[perm]
        filled_matrix = torch.reshape(x, (matrix_.shape))
        return filled_matrix

    def fill_lambdas(self, lambdas_):
        value = lambdas_.shape[0]*lambdas_.shape[1]
        x1 = torch.empty(value)
        x1 = torch.nn.init.trunc_normal_(x1, 0, 0.5)
        x1 = abs(x1)+0.2
        return x1


    def compute_Weight(self, module: Module) -> Any:
        """
        Method to specify how weight is computed for each forward pass

        Args:
            module (Module): Module for which weight is calculated

        Returns:
            Tensor: Computed weight
        """
        # Get the registered weights and sigma
        weights = getattr(module, self.name+'_weights')
        sigma = getattr(module, self.name+'_sigma')
        # Get the factor matrices for Conv layer
        if isinstance(module,torch.nn.Conv2d):
            A = getattr(module, self.name+'_A')
            B = getattr(module, self.name+'_B')
            C = getattr(module, self.name+'_C')
            D = getattr(module, self.name+'_D')
            facs = (weights, [A, B, C, D])
        # Get the factor matrices for linear layer
        elif isinstance(module, torch.nn.Linear):
            A = getattr(module, self.name+'_A')
            B = getattr(module, self.name+'_B')
            facs = (weights, [A, B])
        # Normalize factor matrices
        _, factors = tl.cp_normalize(facs)
        # Multiply sigma to weights for weight calculation
        cp_layer = (weights*(1), factors)
        recons_weight = tl.cp_to_tensor(cp_layer)
        return recons_weight

    @staticmethod
    def apply(module, name: str, rank : int, max_iter : int, init_method):
        """
        This method computes the CP decomposition of the weight tensor
        using CPPower method and registers the resultant parameters

        Args:
            module (module): Module for which CPNorm is calculated
            name (str): Name of the module to apply CP Norm [weight 
                        or bias]
            rank (int): Rank of the tensor for decomposition
            max_iter (int): Maximum iterations for decomposition (used
                            only in case of parafac decomposition)

        Raises:
            RuntimeError: If CPNorm is already applied then error is 
                          raised

        Returns:
            [module]: CPNorm applied module
        """
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, CPNorm) and hook.name == name:
                raise RuntimeError('This parameter already has CP norm \
                                    applied')

        fn = CPNorm(name)

        weight_tensor = getattr(module, name)

        del module._parameters[name]
        # Move the weight tensor to cuda in case not
        if not weight_tensor.is_cuda:
            tensor_ = tl.tensor(weight_tensor, device='cuda:0', 
                                dtype=tl.float32)
        # factors = parafac(tensor_, rank= rank, init='random', 
        #                   n_iter_max = max_iter, 
        #                   normalize_factors = False)
        
        # Calculate factors from CPPower
        if init_method == 'CPD':
            CPP = CPPower(rank = rank)
            factors = CPP.fit_transform(tensor_)
        
        # Register factors for cinvolutional layer
        if isinstance(module, torch.nn.Conv2d):
            A = torch.empty((weight_tensor.shape[0], rank), requires_grad=True)
            B = torch.empty((weight_tensor.shape[1], rank), requires_grad=True)
            C = torch.empty((weight_tensor.shape[2], rank), requires_grad=True)
            D = torch.empty((weight_tensor.shape[3], rank), requires_grad=True)
            if init_method == 'CPD':
                print('####CPD####')
                A, B = factors[1][0].cpu(), factors[1][1].cpu()
                C, D = factors[1][2].cpu(), factors[1][3].cpu() 
            elif init_method == 'KNORMAL':
                print('####KNORMAL####')
                A = torch.nn.init.kaiming_normal_(A)
                B = torch.nn.init.kaiming_normal_(B)
                C = torch.nn.init.kaiming_normal_(C)
                D = torch.nn.init.kaiming_normal_(D)
            elif init_method == 'KUNIFORM':
                print('####KUNIFORM####')
                A = torch.nn.init.kaiming_uniform_(A)
                B = torch.nn.init.kaiming_uniform_(B)
                C = torch.nn.init.kaiming_uniform_(C)
                D = torch.nn.init.kaiming_uniform_(D)
            elif init_method == 'MIXED':
                print('####MIXED####')
                A = torch.nn.init.kaiming_normal_(A)
                B = torch.nn.init.kaiming_normal_(B)
                C = fn.fill_multivariate_normal(C)
                D = fn.fill_multivariate_normal(D)
                
            module.register_parameter(name+'_A', Parameter(A))
            module.register_parameter(name+'_B', Parameter(B))
            module.register_parameter(name+'_C', Parameter(C))
            module.register_parameter(name+'_D', Parameter(D))

        # Register factors for linear layers
        elif isinstance(module, torch.nn.Linear):
            A = torch.empty((weight_tensor.shape[0], rank), requires_grad=True)
            B = torch.empty((weight_tensor.shape[1], rank), requires_grad=True)
            if init_method == 'CPD':
                A, B = factors[1][0].cpu(), factors[1][1].cpu()
            elif init_method == 'KNORMAL':
                A = torch.nn.init.kaiming_normal_(A)
                B = torch.nn.init.kaiming_normal_(B)
            elif init_method == 'KUNIFORM':
                A = torch.nn.init.kaiming_uniform_(A)
                B = torch.nn.init.kaiming_uniform_(B)

            module.register_parameter(name+'_A', Parameter(A))
            module.register_parameter(name+'_B', Parameter(B))

        # Register lambdas and simga for each weight
        module.register_parameter(name+'_sigma',
                                  Parameter(torch.tensor([1],
                                  dtype=torch.float32)))
        if init_method == 'CPD':
            module.register_parameter(name+'_weights',
                                      Parameter(factors[0].cpu()))
        elif init_method == 'MIXED':
            lbds = torch.empty((1, rank), requires_grad=True)
            lbds = fn.fill_lambdas(lbds)
            module.register_parameter(name+'_weights',
                                      Parameter(lbds.cpu()))
        else:
            module.register_parameter(name+'_weights',
                                      Parameter(torch.ones((rank)).cpu()))

        # Specify function to compute weight for forward pass 
        setattr(module, name, fn.compute_Weight(module))
        module.register_forward_pre_hook(fn)

        return fn

    @staticmethod
    def inference_apply(module, name: str, rank : int):
        """
        Method creates the dummy factors of respective shapes in case of
        inference or fine tuning case, where calculating CP decomposition
        additional computation overhead

        Args:
            module (module): Module for which CPNorm is calculated
            name (str): Name of the module to apply CP Norm [weight 
                        or bias]
            rank (int): Rank of the tensor for decomposition

        Raises:
            RuntimeError: If CPNorm is already applied then error is 
                          raised

        Returns:
            [module]: CPNorm applied module
        """
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, CPNorm) and hook.name == name:
                raise RuntimeError('This parameter already has CP norm \
                                    applied')

        fn = CPNorm(name)

        weight_tensor = getattr(module, name)

        del module._parameters[name]
        # Create dummy parameters in case of original parameters
        lambda_ = torch.randn((rank,), requires_grad=True)

        # Register these dummy parameters of same shape
        if isinstance(module, torch.nn.Conv2d):
            A = torch.randn((weight_tensor.shape[0], rank),
                            requires_grad=True)
            B = torch.randn((weight_tensor.shape[1], rank),
                            requires_grad=True)
            C = torch.randn((weight_tensor.shape[2], rank),
                            requires_grad=True)
            D = torch.randn((weight_tensor.shape[3], rank),
                            requires_grad=True)
            module.register_parameter(name+'_A', Parameter(A))
            module.register_parameter(name+'_B', Parameter(B))
            module.register_parameter(name+'_C', Parameter(C))
            module.register_parameter(name+'_D', Parameter(D))

        elif isinstance(module, torch.nn.Linear):
            A = torch.randn((weight_tensor.shape[0], rank),
                            requires_grad=True)
            B = torch.randn((weight_tensor.shape[1], rank),
                            requires_grad=True)
            module.register_parameter(name+'_A', Parameter(A))
            module.register_parameter(name+'_B', Parameter(B))
            
        module.register_parameter(name+'_sigma',
                                  Parameter(torch.tensor([1],
                                  dtype=torch.float32)))
        module.register_parameter(name+'_weights', Parameter(lambda_))
        setattr(module, name, fn.compute_Weight(module))
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        """
        Method to get back original weight tensor back

        Args:
            module (Module): Module for which weight must be calculated
        """
        # Compute the weight tensor
        weight = self.compute_Weight(module)
        # Delete all the factors, lambdas and sigma
        delattr(module, self.name)
        del module._parameters[self.name+'_weights']
        if isinstance(module, torch.nn.Conv2d):
            del module._parameters[self.name+'_A']
            del module._parameters[self.name+'_B']
            del module._parameters[self.name+'_C']
            del module._parameters[self.name+'_D']
        elif isinstance(module, torch.nn.Linear):
            del module._parameters[self.name+'_A']
            del module._parameters[self.name+'_B']

        setattr(module, self.name, Parameter(weight))
    
    def __call__(self, module: Module, inputs : Any) -> None :
        setattr(module, self.name, self.compute_Weight(module))
# End of class

T_module = TypeVar('T_module', bound=Module)
def cp_norm(module : T_module, rank : int = None, name : str = 'weight',
            max_iter : int = 100, inference = False, init_method = 'CPD'):
    """
    Method to apply respective cp norm i.e., training or inference mode

    Args:
        module (T_module): Layer for which normalization must be applied
        rank (int, optional): Full rank of the weight. Defaults to None.
        name (str, optional): name of module to apply CP norm.
                              Defaults to 'weight'.
        max_iter (int, optional): Maximum iterations in case of parafac 
                                  decomposition. Defaults to 100.
        inference (bool, optional): Boolean variable to specify 
                                    inference/trianing mode. 
                                    Defaults to False.

    Returns:
        [T_module]: Layer with CP norm applied
    """
    if rank is None:
        # estimate the tensor rank
        rank = estimate_rank(module.weight)
    # If inference/fine tuning only decoy CP tensors are formed
    if inference:
        CPNorm.inference_apply(module, name, rank)
    # If training mode CP decomposition is estimated
    else:
        CPNorm.apply(module, name, rank, max_iter, init_method)
    return module
