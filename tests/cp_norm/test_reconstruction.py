import torch
from alexnet_model import AlexNet
from tensorly.decomposition import parafac
import tensorly as tl
tl.set_backend('pytorch')
import timeit
import torchvision.models as models
from mnist import Net
import sys

def estimate_rank(tensor: torch.Tensor, max_it: int = 2500 ) -> int:
    """
    This function is used to estimate the rank of the tensor

    Args:
        tensor (torch.Tensor): input tensor
        max_it (int, optional):Maximum iterains. Defaults to 2500.

    Returns:
        int: rank of tensor
    """
    count = 0
    for it in range(1680, max_it, 7):
        #print(it, end=' , ')
        try:
            stime = timeit.timeit()
            # Compute decomposition and reconstruction
            decomposition = parafac(tensor.cpu(), rank=it, init='random', random_state = 0)
            reconstruction = tl.cp_to_tensor(decomposition)
            etime = timeit.timeit()
            reconstruction = reconstruction.to(device)
            # Compute error and fit
            err = torch.mean(torch.abs(reconstruction - tensor))
            fit = torch.linalg.norm(tensor-reconstruction)/torch.linalg.norm(tensor)
            #print(err.item(), ' , ', fit.item(), ' , Time: ', (etime-stime), flush=True)
            time_val = etime-stime
            print('Rank_Try: {0}, Mean_error: {1:5f}, Fit(%): {2:3f}, Time: {3:3f}'.format(it, err.item(), (1-fit.item())*100, time_val), flush=True)
            # if err < torch.finfo(torch.float32).eps:
            if err.item() != err.item():
                count += 1
            if err < 1e-5 or (1-fit.item())*100 > 99.0 or count > 3:
                print()
                print('Convergence reached...', flush=True)
                break
        except Exception as e:
            print(e)
            print('rank', it, 'failed to converge.', flush=True)
    return it

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device, flush=True)

model = AlexNet()
#model = models.resnet18(pretrained=True)
#model = Net()
model = model.to(device)
for index, (name, layer) in enumerate(model.named_modules()):
    #if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
    #if isinstance(layer, torch.nn.Linear) and index == int(sys.argv[1]):
    #    print('name: ', name, '   ', layer.weight.shape, flush=True)
    #    rank = torch.matrix_rank(layer.weight)
    #    print('Rank: ', rank)
    if index == int(sys.argv[1]):# and isinstance(layer, torch.nn.Conv2d):
        print()
        print('Index: ', index, ' Name: ', name, flush=True)
        _ = estimate_rank(layer.weight)

