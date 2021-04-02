"""
Test the cp_norm function that comes with tensorly.
It looks suspicous, but python does return true for
scales==0 in case of very small values.
"""

import torch
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from tensorly.decomposition import parafac
from tensorly.cp_tensor import _validate_cp_tensor
tl.set_backend('pytorch')
from tensorly import backend as T


def fix_cp_normalize(cp_tensor):
    """From 
       http://tensorly.org/stable/_modules/tensorly/cp_tensor.html#cp_normalize
       fixed the division by zero problem.
    """
    _, rank = _validate_cp_tensor(cp_tensor)
    weights, factors = cp_tensor
    
    if weights is None:
        weights = T.ones(rank, **T.context(factors[0]))
    
    normalized_factors = []
    for i, factor in enumerate(factors):
        if i == 0:
            factor = factor*weights
            weights = T.ones(rank, **T.context(factor))
            
        scales = T.norm(factor, axis=0)
        # scales_non_zero = T.where(scales==0, T.ones(T.shape(scales), **T.context(factor)), scales)
        scales_non_zero = T.where(scales < T.finfo(T.float32).eps,
                                  T.ones(T.shape(scales),
                                  **T.context(factor)), scales)
        weights = weights*scales
        normalized_factors.append(factor / T.reshape(scales_non_zero, (1, -1)))

    return CPTensor((weights, normalized_factors))


tensor = torch.rand((2, 2, 2))

factors = parafac(tensor, rank=6)
reconstruction_tensor = tl.cp_to_tensor(factors)
factors.weights[-1] = factors.weights[-1]/1e20
problem_tensor = tl.cp_to_tensor(factors)


norm_factors_fixed = fix_cp_normalize(factors)
norm_factors = tl.cp_normalize(factors)

print('norm', norm_factors.weights, norm_factors.factors)
print('fixed', norm_factors_fixed.weights, norm_factors_fixed.factors)

rec_norm = tl.cp_to_tensor(norm_factors)
rec_fix = tl.cp_to_tensor(norm_factors_fixed)

print('err norm', tl.norm(tl.abs(problem_tensor - rec_norm)))
print('err fixed', tl.norm(tl.abs(problem_tensor - rec_fix)))



print('done')
