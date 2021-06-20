import torch

def computePsi1_numerical_problems(scale, lengthscales, variational_mean, variational_var, inducing_variable):
    lengthscales2 = torch.square(lengthscales)
    # lenthscales # D
    Xmu = variational_mean # NxD
    Xcov = variational_var # NxD
    Z = torch.transpose(inducing_variable, 0, 1) # DxM
    Xmu_u = torch.unsqueeze(Xmu, 2) # NxDx1
    # Xmu_u = np.expand_dims(Xmu, 2) # NxDx1
    all_diffs = Z - Xmu_u  # NxDxM
    Xcov_denom = lengthscales2 + Xcov  # NxD
    Xcov_denom_u = torch.unsqueeze(Xcov_denom, 2) # NxDx1
    # Xcov_denom_u = np.expand_dims(Xcov_denom, 2) # NxDx1
    exponent_mahalanobis = -0.5*all_diffs**2/Xcov_denom_u # NxDxM
    exponential_arg = torch.sum(exponent_mahalanobis, 1) # NxM
    exponential_value = torch.exp(exponential_arg) # NxM
    psi1_denominator = torch.sqrt(Xcov_denom/lengthscales2) # NxM
    psi1_denominator_prod = torch.prod(psi1_denominator, 1) # N
    psi1_denominator_prod_u = torch.unsqueeze(psi1_denominator_prod, 1) # Nx1
    # psi1_denominator_prod_u = np.expand_dims(psi1_denominator_prod, 1) # Nx1
    psi1 = scale*exponential_value/psi1_denominator_prod_u # NxM
    return psi1

def computePsi1(scale, lengthscales, variational_mean, variational_var, inducing_variable):
    variance = scale
    lengthscale = lengthscales
    Z = inducing_variable
    mu = variational_mean
    S = variational_var

    # def __psi1computations(variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi1
    # Produced intermediate results:
    # _psi1                NxM

    lengthscale2 = torch.square(lengthscale)

    # psi1
    _psi1_logdenom = torch.log(S/lengthscale2+1.).sum(axis=-1) # N
    _psi1_log = (_psi1_logdenom[:,None]+torch.einsum('nmq,nq->nm',torch.square(mu[:,None,:]-Z[None,:,:]),1./(S+lengthscale2)))/(-2.)
    _psi1 = variance*torch.exp(_psi1_log)

    return _psi1


def computePsi2_numerical_problems(scale, lengthscales, variational_mean, variational_var, inducing_variable):
    lengthscales2 = torch.square(lengthscales)
    # lenthscales # D
    lengthscales2_div2 = lengthscales2*0.5 # 1xD
    Xmu = variational_mean # NxD
    Xcov = variational_var # NxD
    Z = inducing_variable # MxD
    Zt = torch.transpose(Z, 0, 1) # DxM
    Z_u = torch.unsqueeze(Z, 2) # MxDx1
    # Z_u = np.expand_dims(Z, 2) # MxDx1
    all_Zdiffs = Zt - Z_u  # MxDxM
    lengthscales2_div2_u = torch.unsqueeze(lengthscales2_div2, 2) # 1xDx1
    # lengthscales2_div2_u = np.expand_dims(lengthscales2_div2, 1) # Dx1
    exponent_mahalanobisZ = all_Zdiffs**2/lengthscales2_div2_u # MxDxM
    exponential_argZ = -0.125*torch.sum(exponent_mahalanobisZ, 1) # MxM
    exponential_valueZ = torch.exp(exponential_argZ) # MxM

    all_Zavg = 0.5*(Zt + Z_u)  # MxDxM
    Xmu_u = torch.unsqueeze(Xmu, 1) # NX1XD
    Xmu_u = torch.unsqueeze(Xmu_u, 3) # NX1XDx1
    # Xmu_u = np.expand_dims(Xmu, 1) # NX1XD
    # Xmu_u = np.expand_dims(Xmu_u, 3) # NX1XDx1
    all_XmuZdiffs = Xmu_u - all_Zavg  # NxMxDxM
    Xcov_denom = Xcov+lengthscales2_div2 # NxD
    Xcov_denom_u = torch.unsqueeze(Xcov_denom, 1) # Nx1xD
    Xcov_denom_u = torch.unsqueeze(Xcov_denom_u, 3) # Nx1xDx1
    # Xcov_denom_u = np.expand_dims(Xcov_denom, 1) # Nx1xD
    # Xcov_denom_u = np.expand_dims(Xcov_denom_u, 3) # Nx1xDx1
    exponent_mahalanobisXmuZ = all_XmuZdiffs**2/Xcov_denom_u # NxMxDxM
    exponential_argXmuZ = -0.5*torch.sum(exponent_mahalanobisXmuZ, 2) # NxMxM
    exponential_valueXmuZ = torch.exp(exponential_argXmuZ) # NxMxM

    psi2_denominator = torch.sqrt(Xcov_denom/lengthscales2_div2) # NxD
    psi2_denominator_prod = torch.prod(psi2_denominator, 1) # N
    psi2_denominator_prod_u = torch.unsqueeze(psi2_denominator_prod, 1) # Nx1
    psi2_denominator_prod_u = torch.unsqueeze(psi2_denominator_prod_u, 1) # Nx1x1
    # psi2_denominator_prod_u = np.expand_dims(psi2_denominator_prod, 1) # Nx1
    # psi2_denominator_prod_u = np.expand_dims(psi2_denominator_prod_u, 1) # Nx1x1

    psi2_notSummed = scale**2*exponential_valueZ*exponential_valueXmuZ/psi2_denominator_prod_u # NxMxM
    psi2 = torch.sum(psi2_notSummed, 0)
    return psi2


def computePsi2(scale, lengthscales, variational_mean, variational_var, inducing_variable):
    variance = scale
    lengthscale = lengthscales
    Z = inducing_variable
    mu = variational_mean
    S = variational_var
    # def __psi2computations(variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi2
    # Produced intermediate results:
    # _psi2                MxM

    N,M,Q = mu.shape[0], Z.shape[0], mu.shape[1]
    lengthscale2 = torch.square(lengthscale)

    _psi2_logdenom = torch.log(2.*S/lengthscale2+1.).sum(axis=-1)/(-2.) # N
    _psi2_exp1 = (torch.square(Z[:,None,:]-Z[None,:,:])/lengthscale2).sum(axis=-1)/(-4.) #MxM
    Z_hat = (Z[:,None,:]+Z[None,:,:])/2. #MxMxQ
    denom = 1./(2.*S+lengthscale2)
    _psi2_exp2 = -(torch.square(mu)*denom).sum(axis=-1)[:,None,None]+(2*(mu*denom).matmul(Z_hat.reshape(M*M,Q).T) - denom.matmul(torch.square(Z_hat).reshape(M*M,Q).T)).reshape(N,M,M)
    _psi2 = variance*variance*torch.exp(_psi2_logdenom[:,None,None]+_psi2_exp1[None,:,:]+_psi2_exp2)
    answer = torch.sum(_psi2, 0)
    return answer

