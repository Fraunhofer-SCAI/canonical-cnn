import torch

def compress_via_reparam(layer, compress_rate):
    """Apply compression by removing the lambdas and corresponding factor
    matrix vectors. Compress based on the compression rate.

    Args:
        layer (Conv2d/Linear): Either torch Conv layer or Linear layer
        compress_rate (int): Specified compression rate

    Returns:
        Conv2d/Linear: Either reparameterized Conv or Linear layer
    """
    # If conv layer get four factor matrices because it is 4D tensor
    if isinstance(layer, torch.nn.Conv2d):
        lmbds = layer.weight_weights
        # Calculate the number of lambds required for compression
        k = int(len(lmbds)*(1-(compress_rate/100)))
        # Read all the parameters of corresponding layer
        factor_A, factor_B = layer.weight_A, layer.weight_B
        factor_C, factor_D = layer.weight_C, layer.weight_D
        # Sort lambds in descending order & get top k lambdas and vectors
        lmbds_sorted, indices = torch.sort(lmbds, descending=True)
        lmbds_sorted, indices = lmbds_sorted[0:k], indices[0:k]
        factor_A, factor_B = factor_A[:, indices], factor_B[:, indices]
        factor_C, factor_D = factor_C[:, indices], factor_D[:, indices]
        # Instantiate with new parameters
        layer.weight_weights = torch.nn.Parameter(lmbds_sorted)
        layer.weight_A, layer.weight_B = torch.nn.Parameter(factor_A), torch.nn.Parameter(factor_B)
        layer.weight_C, layer.weight_D = torch.nn.Parameter(factor_C), torch.nn.Parameter(factor_D)

    # If linear layer get two factor matrices becuase it is 2D tensor
    if isinstance(layer, torch.nn.Linear):
        lmbds = layer.weight_weights
        # Calculate the number of lambds required for compression
        k = int(len(lmbds)*(1-(compress_rate/100)))
        # Read all the parameters of corresponding layer
        factor_A, factor_B = layer.weight_A, layer.weight_B
        # Sort lambds in descending order & get top k lambdas and vectors
        lmbds_sorted, indices = torch.sort(lmbds, descending=True)
        lmbds_sorted, indices = lmbds_sorted[0:k], indices[0:k]
        factor_A, factor_B = factor_A[:, indices], factor_B[:, indices]
        # Instantiate with new parameters
        layer.weight_weights = torch.nn.Parameter(lmbds_sorted)
        layer.weight_A, layer.weight_B = torch.nn.Parameter(factor_A), torch.nn.Parameter(factor_B)
    
    return layer


def apply_compression(model, compress_rate):
    """For each layer in model apply given compression rate

    Args:
        model : Neural network model
        compress_rate (int): Specified compression rate

    Returns:
        : Neural network model
    """
    # Iterate over each layer of the model and compress Conv and Linear
    for index, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, torch.nn.Conv2d): #or isinstance(layer, torch.nn.Linear):
            layer = compress_via_reparam(layer, compress_rate)
    return model

if __name__ == '__main__':
    apply_compression(model, compress_rate)