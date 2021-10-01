import json
import torch
import torch.nn as nn

def apply_lowrank_weights(W, b, conv_v, conv_h):
  num_grps = conv_v.weight.shape[0]//conv_h.weight.shape[1]
  N, C, d1, d2 = W.shape
  K = conv_h.weight.shape[1]
  N = N//num_grps
  for idx in range(num_grps):
    # Bijection:Tensor->Matrix
    w_matrix = W[N*idx : N*(idx+1)].permute(1, 2, 3, 0).reshape((C*d1, d2*N))
    # SVD
    U, S, V = torch.linalg.svd(w_matrix, full_matrices=False)
    # v weight

    v = U[:, :K]*torch.sqrt(S[:K])
    v = v[:, :K].reshape((C, d1, 1, K)).permute(3, 0, 1, 2)
    # h weight
    h = V[:K, :] * torch.unsqueeze(torch.sqrt(S)[:K], 1)
    h = h.reshape((K, 1, d2, N)).permute(3, 0, 1, 2)
    # Apply weights to weight matrix
    with torch.no_grad():
      conv_v.weight[K*idx : K*(idx+1)] = v
      conv_h.weight[N*idx : N*(idx+1)] = h
    return conv_v, conv_h

def tai_decompose(model, config_file):
    low_rank_config = None
    with open(config_file) as json_file:
        low_rank_config = json.load(json_file)
    if low_rank_config == None:
        raise RuntimeError("Empty config file")
    print()
    print('Config file: ')
    print(low_rank_config)
    print(model.eval())
    for key, value in low_rank_config.items():
        for idx, (name, layer) in enumerate(model.named_modules()):
            if (key in name) and (isinstance(layer, nn.Conv2d)):
                W, b = layer.weight, layer.bias
                N, C, d1, d2 = W.shape
                S1, S2 = layer.stride
                P1, P2 = layer.padding
                groups = layer.groups
                convlayer_v = nn.Conv2d(C, value, (d1, 1), padding=(P1, 0),
                                        stride=(S1, 1), groups = groups)
                convlayer_h = nn.Conv2d(value, N, (1, d1),  padding=(0, P2),
                                        stride=(1, S2), groups= groups)
                convlayer_v, convlayer_h = apply_lowrank_weights(W, b, 
                                                    convlayer_v, convlayer_h)
                new_convlayers = nn.Sequential(convlayer_v, 
                                                convlayer_h)
                #setattr(model, name, new_convlayers)
                model.features[int(key.split('.')[-1])] = new_convlayers
    return model



