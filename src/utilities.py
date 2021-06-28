import numpy as np
import torch

def flatten_params(params_iter):
    params_names = []
    flattened = np.array([])
    for param_name, param_value in params_iter:
        params_names.append(param_name)
        flattened = np.concatenate((flattened, param_value.detach().numpy().flatten()))
    return params_names, flattened

def set_model_params(flattened_params, param_names, model):
    params = model.parameters()
    index = 0
    for param_name in param_names:
        sep_index = param_name.find(".")
        if sep_index<0:
            # root object
            param = getattr(model, param_name)
            param_length = param.numel()
            param_shape = param.shape
            to_set_flattened_param = flattened_params[index:(index+param_length)]
            to_set_param = torch.nn.Parameter(torch.from_numpy(to_set_flattened_param.reshape(param.shape)), requires_grad=True)
            setattr(model, param_name, to_set_param)
        else:
            child_object_name = param_name[:sep_index]
            child_object_param_name = param_name[sep_index+1:]
            child_object = getattr(model, child_object_name)
            sep_index = child_object_param_name.find(".")
            if sep_index<0:
                # child object
                param = getattr(child_object, child_object_param_name)
                param_length = param.numel()
                param_shape = param.shape
                to_set_flattened_param = flattened_params[index:(index+param_length)]
                to_set_param = torch.nn.Parameter(torch.from_numpy(to_set_flattened_param.reshape(param.shape)), requires_grad=True)
                setattr(child_object, child_object_param_name, to_set_param)
            else:
                grandchild_object_name = child_object_param_name[:sep_index]
                grandchild_object_param_name = child_object_param_name[sep_index+1:]
                grandchild_object = getattr(child_object, grandchild_object_name)
                sep_index = grandchild_object_param_name.find(".")
                if sep_index<0:
                    # grandchild object
                    param = getattr(grandchild_object, grandchild_object_param_name)
                    param_length = param.numel()
                    param_shape = param.shape
                    to_set_flattened_param = flattened_params[index:(index+param_length)]
                    to_set_param = torch.nn.Parameter(torch.from_numpy(to_set_flattened_param.reshape(param.shape)), requires_grad=True)
                    setattr(grandchild_object, grandchild_object_param_name, to_set_param)
                else:
                    raise RuntimeError("Too many layers of objects")
        index += param_length
        
def get_model_grad(param_names, model):
    grad = np.array([], dtype=np.double)
    params = model.parameters()
    for param_name in param_names:
        sep_index = param_name.find(".")
        if sep_index<0:
            # root object
            param = getattr(model, param_name)
            grad = np.concatenate((grad, param.grad.flatten()))
        else:
            child_object_name = param_name[:sep_index]
            child_object_param_name = param_name[sep_index+1:]
            child_object = getattr(model, child_object_name)
            sep_index = child_object_param_name.find(".")
            if sep_index<0:
                # child object
                param = getattr(child_object, child_object_param_name)
                grad = np.concatenate((grad, param.grad.flatten()))
            else:
                grandchild_object_name = child_object_param_name[:sep_index]
                grandchild_object_param_name = child_object_param_name[sep_index+1:]
                grandchild_object = getattr(child_object, grandchild_object_name)
                sep_index = grandchild_object_param_name.find(".")
                if sep_index<0:
                    param = getattr(grandchild_object, grandchild_object_param_name)
                    grad = np.concatenate((grad, param.grad.flatten()))
                else:
                    raise RuntimeError("Too many layers of objects")
    return grad

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

