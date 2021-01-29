from torch.nn.parameter import Parameter#, UninitializedParameter
from typing import Any, TypeVar
from torch.nn import Module
import tensorly as tl
import torch
tl.set_backend('pytorch')
from tensorly.decomposition import parafac

class CPNorm(object):
    name : str

    def __init__(self, name : str) -> None:
        self.name = name

    def compute_Weight(self, module: Module) -> Any:
        A = getattr(module, self.name+'_A')
        B = getattr(module, self.name+'_B')
        C = getattr(module, self.name+'_C')
        D = getattr(module, self.name+'_D')
        weights = getattr(module, self.name+'_weights')
        recons_weight = tl.cp_to_tensor((weights, [A, B, C, D]))
        return recons_weight
    
    @staticmethod
    def apply(module, name: str, rank : int, max_iter : int):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, CPNorm) and hook.name == name:
                raise RuntimeError('This parameter already has CP norm applied')
        
        fn = CPNorm(name)

        weight_tensor = getattr(module, name)

        #if isinstance(weight_tensor, UninitializedParameter):
        #    raise ValueError(
        #        'Underconstruction, needed more clarity regarding this error')
        
        del module._parameters[name]

        factors = parafac(weight_tensor, rank= rank, init='random', random_state = 0,n_iter_max = max_iter, normalize_factors = True)
        lbda = factors[0]
        A, B, C, D = factors[1][0], factors[1][1], factors[1][2], factors[1][3] 
        module.register_parameter(name+'_A', Parameter(A))
        module.register_parameter(name+'_B', Parameter(B))
        module.register_parameter(name+'_C', Parameter(C))
        module.register_parameter(name+'_D', Parameter(D))
        module.register_parameter(name+'_weights', Parameter(factors[0]))
        setattr(module, name, fn.compute_Weight(module))

        module.register_forward_pre_hook(fn)

        return fn
    
    def remove(self, module: Module) -> None:
        weight = self.compute_Weight(module)
        delattr(module, self.name)
        del module._parameters[self.name+'_lbda']
        del module._parameters[self.name+'_A']
        del module._parameters[self.name+'_B']
        del module._parameters[self.name+'_C']
        del module._parameters[self.name+'_D']
        setattr(module, self.name, Parameter(weight))
    
    def __call__(self, module: Module, inputs : Any) -> None :
        setattr(module, self.name, self.compute_Weight(module))
# End of class

T_module = TypeVar('T_module', bound=Module)
def CP_Norm(module : T_module, rank : int, name : str = 'weight', max_iter : int = 100) -> T_module:
    CPNorm.apply(module, name, rank, max_iter)
            
    
