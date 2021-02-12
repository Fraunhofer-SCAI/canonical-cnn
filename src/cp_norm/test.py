import torch
from alexnet_model import AlexNet
from tensorly.decomposition import parafac
import tensorly as tl
tl.set_backend('pytorch')
import timeit
import torchvision.models as models
from mnist import Net

def estimate_rank(tensor: torch.Tensor, max_it: int = 1500 ) -> int:
    count = 0
    for it in range(1, max_it, 5):
        #print(it, end=' , ')
        try:
            stime = timeit.timeit()
            decomposition = parafac(tensor.cpu(), rank=it, init='random', random_state = 0)
            reconstruction = tl.cp_to_tensor(decomposition)
            etime = timeit.timeit()
            reconstruction = reconstruction.to(device)
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
                print('Convergence reached...')
                break
        except Exception as e:
            print(e)
            print('rank', it, 'failed to converge.')
    return it

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device, flush=True)

model = AlexNet()
#model = models.resnet18(pretrained=True)
#model = Net()
model = model.to(device)
for index, (name, layer) in enumerate(model.named_modules()):
    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
        print()
        print('Index: ', index, ' Name: ', name)
        _ = estimate_rank(layer.weight)

