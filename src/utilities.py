import numpy as np
import torch
import gpytorch

def computePsi1(scale, lengthscales, variational_mean, variational_var, inducing_variable):
    lengthscales2 = torch.square(lengthscales)

    psi1_logdenom = torch.log(variational_var/lengthscales2+1.).sum(axis=-1) # N
    psi1_log = (psi1_logdenom[:,None]+torch.einsum('nmq,nq->nm',torch.square(variational_mean[:,None,:]-inducing_variable[None,:,:]),1./(variational_var+lengthscales2)))/(-2.)
    psi1 = scale*torch.exp(psi1_log)

    return psi1

def computePsi2(scale, lengthscales, variational_mean, variational_var, inducing_variable):
    # variational_mean, variational_var \in NxQ
    # inducing_variable \in MxQ
    N,M,Q = variational_mean.shape[0], inducing_variable.shape[0], variational_mean.shape[1]
    lengthscales2 = torch.square(lengthscales)

    psi2_logdenom = torch.log(2.*variational_var/lengthscales2+1.).sum(axis=-1)/(-2.) # N
    psi2_exp1 = (torch.square(inducing_variable[:,None,:]-inducing_variable[None,:,:])/lengthscales2).sum(axis=-1)/(-4.) #MxM
    Z_hat = (inducing_variable[:,None,:]+inducing_variable[None,:,:])/2. #MxMxQ
    denom = 1./(2.*variational_var+lengthscales2) #NxQ
    #comments for the next two lines
    #Nx1x1
    #NxQ%*%QxM^2=NxM^2 
    psi2_exp2 = -(torch.square(variational_mean)*denom).sum(axis=-1)[:,None,None]+\
                (2*(variational_mean*denom).matmul(Z_hat.reshape(M*M,Q).T)-\
                 denom.matmul(torch.square(Z_hat).reshape(M*M,Q).T)).reshape(N,M,M) #NxQ%*%QxM^2=NxM^2
    psi2 = scale*scale*torch.exp(psi2_logdenom[:,None,None]+psi2_exp1[None,:,:]+psi2_exp2)
    answer = torch.sum(psi2, 0)
    return answer

def simulate(N, D, Q, output_scale, lengthscales, likelihood_var):
    X = torch.reshape(input=torch.normal(mean=0, std=torch.ones(N*Q)),
                      shape=(N,Q))
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=Q))
    kernel.outputscale = output_scale
    kernel.base_kernel.lengthscale = lengthscales
    K_NN = kernel(X).evaluate()
    I_N = torch.eye(N, dtype=torch.double)
    Y = torch.empty(size=(N, D), dtype=torch.double)
    mean = torch.zeros(N, dtype=torch.double)
    cov = K_NN+likelihood_var*I_N
    mvn = torch.distributions.MultivariateNormal(mean, cov)
    for d in range(D):
        Y[:,d] = mvn.sample()
    return Y, X
