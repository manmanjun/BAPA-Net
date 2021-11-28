import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.special import erfinv
import torch
import torch.nn.functional as F

def generate_cutout_mask(img_size, seed = None):
    np.random.seed(seed)

    cutout_area = img_size[0] * img_size[1] / 2

    w = np.random.randint(img_size[1] / 2, img_size[1] + 1)
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = np.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.astype(float)
'''
def generate_bernoulli_mask(img_size, sigma, p, seed=None):
    np.random.seed(seed)
    # Randomly draw sigma from log-uniform distribution
    N = np.random.normal(size=img_size) # Generate noise image
    Ns = N
    #Ns = gaussian_filter(N, sigma) # Smooth with a Gaussian
    # Compute threshold
    t = erfinv(p*2 - 1) * (2**0.5) * Ns.std() + Ns.mean()
    return (Ns > t).astype(float) # Apply threshold and return'''

def generate_cow_mask(img_size, sigma, p, seed=None):
    np.random.seed(seed)
    # Randomly draw sigma from log-uniform distribution
    N = np.random.normal(size=img_size) # Generate noise image
    Ns = gaussian_filter(N, sigma) # Smooth with a Gaussian
    # Compute threshold
    t = erfinv(p*2 - 1) * (2**0.5) * Ns.std() + Ns.mean()
    return (Ns > t).astype(float) # Apply threshold and return
'''
def generate_cloud_mask(img_size, sigma, p,seed=None):
    T=10
    np.random.seed(seed)
    # Randomly draw sigma from log-uniform distribution
    N = np.random.normal(size=img_size) # Generate noise image
    Ns = gaussian_filter(N, sigma) # Smooth with a Gaussian
    Ns_norm = (Ns-Ns.mean())/Ns.std()
    Ns_sharp = np.tanh(T*Ns_norm)
    Ns_normalised = (Ns_sharp - np.min(Ns_sharp))/np.ptp(Ns_sharp)
    return Ns_normalised'''

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    n = 7
    edge_mask = N.clone().detach().double().unsqueeze(0).unsqueeze(0)
    weight_fuui = torch.Tensor(np.ones((n, n))).cuda().double().unsqueeze(0).unsqueeze(0)
    weight_kovh = torch.Tensor(np.ones((n, n))).cuda().double().unsqueeze(0).unsqueeze(0)
    fuui = edge_mask - torch.ge(F.conv2d(edge_mask, weight_fuui, padding=(n-1)//2), n*n).double()
    kovh = torch.ge(F.conv2d(edge_mask, weight_kovh, padding=(n-1)//2), 1).double() - edge_mask
    edge_mask = fuui + kovh #abandon way 
    return N
'''
def generate_cow_class_mask(pred, classes, sigma, p,):
    N=np.zeros(pred.shape)
    pred = np.array(pred.cpu())
    for c in classes:
        N[pred==c] = generate_cow_mask(pred.shape,sigma,p)[pred==c]
    return N'''



def generate_n_n_mask(image_size, classes, n):
    mask = np.array([[0] * n] * n)
    for i in classes:
        mask[i // n][i % n] = 1
    if image_size[0] % n == 0:
        mask = mask.repeat(image_size[0] // n, axis=0).repeat(image_size[1] // n, axis=1)
    else:
        mask = mask.repeat(image_size[0] // n + 1, axis=0).repeat(image_size[1] // n + 1, axis=1)[:image_size[0],:image_size[1]]
    mask = torch.from_numpy(mask).cuda()
    num = 7
    edge_mask = mask.clone().detach().double().unsqueeze(0).unsqueeze(0)
    weight_fuui = torch.Tensor(np.ones((num, num))).cuda().double().unsqueeze(0).unsqueeze(0)
    weight_kovh = torch.Tensor(np.ones((num, num))).cuda().double().unsqueeze(0).unsqueeze(0)
    fuui = edge_mask - torch.ge(F.conv2d(edge_mask, weight_fuui, padding=(num-1)//2), n*n).double()
    kovh = torch.ge(F.conv2d(edge_mask, weight_kovh, padding=(num-1)//2), 1).double() - edge_mask
    edge_mask = fuui + kovh
    return mask, edge_mask


def generate_cutmix_mask(img_size,n):
    mask = np.ones(img_size)
    cutout_area = img_size[0] * img_size[1] / 2
    for i in range(n):
        w = np.random.randint(img_size[1] / 2, img_size[1] + 1)
        h = np.round(cutout_area / w)

        x_start = np.random.randint(0, img_size[1] - w + 1)
        y_start = np.random.randint(0, img_size[0] - h + 1)

        x_end = int(x_start + w)
        y_end = int(y_start + h)
        mask[y_start:y_end, x_start:x_end] = mask[y_start:y_end, x_start:x_end] + 1
    mask[mask%2 == 1] = 1
    mask[mask%2 == 0] = 0
    return mask.astype(float)