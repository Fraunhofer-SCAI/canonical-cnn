from torch.nn.parameter import Parameter#, UninitializedParameter
from typing import Any, TypeVar
from torch.nn import Module
import tensorly as tl
import torch
tl.set_backend('pytorch')
from tensorly.decomposition import parafac, CPPower


def estimate_rank(tensor: torch.Tensor, max_it: int = 1000 ) -> int:
    for it in range(1, max_it, 2):
        try:
            decomposition = parafac(tensor, rank=it, init='random', random_state = 0)
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
    name : str

    def __init__(self, name : str) -> None:
        self.name = name

    def compute_Weight(self, module: Module) -> Any:

        weights = getattr(module, self.name+'_weights')
        sigma = getattr(module, self.name+'_sigma')
        if isinstance(module,torch.nn.Conv2d):
            A = getattr(module, self.name+'_A')
            B = getattr(module, self.name+'_B')
            C = getattr(module, self.name+'_C')
            D = getattr(module, self.name+'_D')
            facs = (weights, [A, B, C, D])
        elif isinstance(module, torch.nn.Linear):
            A = getattr(module, self.name+'_A')
            B = getattr(module, self.name+'_B')
            facs = (weights, [A, B])
        _, factors = tl.cp_normalize(facs)
        cp_layer = (weights*(sigma), factors)
        recons_weight = tl.cp_to_tensor(cp_layer)
        return recons_weight
    
    @staticmethod
    def apply(module, name: str, rank : int, max_iter : int):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, CPNorm) and hook.name == name:
                raise RuntimeError('This parameter already has CP norm applied')

        fn = CPNorm(name)

        weight_tensor = getattr(module, name)

        del module._parameters[name]

        #factors = parafac(weight_tensor, rank= rank, init='random', random_state = 0, 
        #                  n_iter_max = max_iter, normalize_factors = False)
        CPP = CPPower(rank = rank)
        factors = CPP.fit_transform(weight_tensor)
        
        if isinstance(module, torch.nn.Conv2d):
            A, B, C, D = factors[1][0], factors[1][1], factors[1][2], factors[1][3] 
            module.register_parameter(name+'_A', Parameter(A))
            module.register_parameter(name+'_B', Parameter(B))
            module.register_parameter(name+'_C', Parameter(C))
            module.register_parameter(name+'_D', Parameter(D))

        elif isinstance(module, torch.nn.Linear):
            A, B = factors[1][0], factors[1][1]
            module.register_parameter(name+'_A', Parameter(A))
            module.register_parameter(name+'_B', Parameter(B))
        module.register_parameter(name+'_sigma', Parameter(torch.tensor([1], dtype=torch.float32)))
        module.register_parameter(name+'_weights', Parameter(factors[0]))
        setattr(module, name, fn.compute_Weight(module))
        module.register_forward_pre_hook(fn)

        return fn

    @staticmethod
    def inference_apply(module, name: str, rank : int, max_iter : int):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, CPNorm) and hook.name == name:
                raise RuntimeError('This parameter already has CP norm applied')

        fn = CPNorm(name)

        weight_tensor = getattr(module, name)

        del module._parameters[name]

        #factors = parafac(weight_tensor, rank= rank, init='random', random_state = 0, 
        #                  n_iter_max = max_iter, normalize_factors = False)
        #CPP = CPPower(rank = rank)
        #factors = CPP.fit_transform(weight_tensor)
        
        lambda_ = torch.randn((rank,), requires_grad=True)
        if isinstance(module, torch.nn.Conv2d):
            #A, B, C, D = factors[1][0], factors[1][1], factors[1][2], factors[1][3] 
            A = torch.randn((weight_tensor.shape[0], rank), requires_grad=True)
            B = torch.randn((weight_tensor.shape[1], rank), requires_grad=True)
            C = torch.randn((weight_tensor.shape[2], rank), requires_grad=True)
            D = torch.randn((weight_tensor.shape[3], rank), requires_grad=True)
            module.register_parameter(name+'_A', Parameter(A))
            module.register_parameter(name+'_B', Parameter(B))
            module.register_parameter(name+'_C', Parameter(C))
            module.register_parameter(name+'_D', Parameter(D))

        elif isinstance(module, torch.nn.Linear):
            #A, B = factors[1][0], factors[1][1]
            A = torch.randn((weight_tensor.shape[0], rank), requires_grad=True)
            B = torch.randn((weight_tensor.shape[1], rank), requires_grad=True)
            module.register_parameter(name+'_A', Parameter(A))
            module.register_parameter(name+'_B', Parameter(B))
            
        module.register_parameter(name+'_sigma', Parameter(torch.tensor([1], dtype=torch.float32, requires_grad=True)))
        module.register_parameter(name+'_weights', Parameter(lambda_))
        setattr(module, name, fn.compute_Weight(module))
        module.register_forward_pre_hook(fn)

        return fn
    def remove(self, module: Module) -> None:
        weight = self.compute_Weight(module)
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
def cp_norm(module : T_module, rank : int = None, name : str = 'weight', max_iter : int = 100, inference = False):
    if rank is None:
        # estimate the tensor rank
        rank = estimate_rank(module.weight)
    if inference:
        CPNorm.inference_apply(module, name, rank, max_iter)
    else:
        CPNorm.apply(module, name, rank, max_iter)
    return module
