import torch
import torch.nn as nn
import torch.nn.functional as F

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    # TODO: Implement the PGD attack
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool
    
    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()
    
    # If rand_start is True, add uniform noise to the sample within [-eps,+eps],
    # else just copy x_nat
    if(rand_start):
        dat = dat + 2*eps*torch.rand(x_nat.size()).to(device) - eps
    # Make sure the sample is projected into original distribution bounds [0,1]
    
    # Iterate over iters
    for i in range(iters):
        # Compute gradient w.r.t. data
        grad = gradient_wrt_data(model,device,dat,lbl)
        # Perturb the data using the gradient 
        delta = alpha*torch.sign(grad)
        # Clip the perturbed datapoints to ensure we still satisfy L_infinity constraint
        delta = torch.clamp(delta,-eps,eps)
        # Clip the perturbed datapoints to ensure we are in bounds [0,1]
        dat = torch.clamp(x_nat+delta,0,1)
    # Return the final perturbed samples
    return dat


def FGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float
    
    # HINT: FGSM is a special case of PGD
    
    return PGD_attack(model,device,dat,lbl,eps,1,1,False)


def FGM_L2_attack(model, device, dat, lbl, eps):
    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()
    
    # Compute gradient w.r.t. data
    grad = gradient_wrt_data(model,device,dat,lbl)
    # Compute sample-wise L2 norm of gradient (L2 norm for each batch element)
    # HINT: Flatten gradient tensor first, then compute L2 norm
    L2 = dat.flatten().norm(p=2)
#     print(dat.flatten().norm(p=2),dat.flatten().norm(dim=-1,p=2))
#     input()
    # Perturb the data using the gradient
    # HINT: Before normalizing the gradient by its L2 norm, use
    # torch.clamp(l2_of_grad, min=1e-12) to prevent division by 0
    torch.clamp(L2,min=1e-12)
    # Add perturbation the data
    dat = dat+eps*grad/L2
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    dat = torch.clamp(dat,0,1)
    # Return the perturbed samples
    return dat


def random_noise_attack(model, device, dat, eps):
    # Add uniform random noise in [-eps,+eps]
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_adv = torch.clamp(x_adv.clone().detach(), 0., 1.)
    # Return perturbed samples
    return x_adv