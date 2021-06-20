import numpy as np

def computePsi1_simple(scale, lengthscales, variational_mean, variational_var, inducing_variable):
    alphas = 1.0/lengthscales
    N = variational_mean.shape[0]
    M = inducing_variable.shape[0]
    Q = inducing_variable.shape[1]
    psi1 = np.empty((N,M), dtype=np.double)
    for n in range(N):
        for m in range(M):
            psi1[n,m] = scale
            for q in range(Q):
                denominator = alphas[q]*variational_var[n,q]+1
                exp_value = np.exp(-0.5*alphas[q]*(variational_mean[n,q]-inducing_variable[m,q])**2/denominator)
                psi1[n,m] *= exp_value/np.sqrt(denominator)
    return psi1

def computePsi1(scale, lengthscales, variational_mean, variational_var, inducing_variable):
    # lenthscales # D
    Xmu = variational_mean # NxD
    Xcov = variational_var # NxD
    Z = np.transpose(inducing_variable) # DxM
    Xmu_u = np.expand_dims(Xmu, 2) # NxDx1
    all_diffs = Z - Xmu_u  # NxDxM
    Xcov_denom = lengthscales + Xcov  # NxD
    Xcov_denom_u = np.expand_dims(Xcov_denom, 2) # NxDx1
    exponent_mahalanobis = -0.5*all_diffs**2/Xcov_denom_u # NxDxM
    exponential_arg = np.sum(exponent_mahalanobis, 1) # NxM
    exponential_value = np.exp(exponential_arg) # NxM
    psi1_denominator = np.sqrt(Xcov_denom/lengthscales) # NxM
    psi1_denominator_prod = np.prod(psi1_denominator, 1) # N
    psi1_denominator_prod_u = np.expand_dims(psi1_denominator_prod, 1) # Nx1
    psi1 = scale*exponential_value/psi1_denominator_prod_u # NxM
    return psi1

def computePsi2_simple(scale, lengthscales, variational_mean, variational_var, inducing_variable):
    def computePsi2n(n, scale, alphas, variational_mean, variational_var, inducing_variable):
        M = inducing_variable.shape[0]
        Q = inducing_variable.shape[1]
        psi2n = np.empty((M,M), dtype=np.double)
        for m1 in range(M):
            for m2 in range(M):
                psi2n[m1,m2] = scale**2
                for q in range(Q):
                    inducing_variable_mean_q = (inducing_variable[m1,q]+inducing_variable[m2,q])/2.0
                    denominator = 2*alphas[q]*variational_var[n,q]+1
                    exp_value = np.exp(-alphas[q]*(inducing_variable[m1,q]-inducing_variable[m2,q])**2/4.0-
                                        alphas[q]*(variational_mean[n,q]-inducing_variable_mean_q)**2/denominator)
                    psi2n[m1,m2] *= exp_value/np.sqrt(denominator)
        return psi2n

    alphas = 1.0/lengthscales
    N = variational_mean.shape[0]
    M = inducing_variable.shape[0]
    psi2 = np.zeros((M, M), dtype=np.double)
    for n in range(N):
        print("Processing Psi2[{:d}]".format(n))
        psi2 += computePsi2n(n=n, scale=scale, alphas=alphas,
                             variational_mean=variational_mean, variational_var=variational_var,
                             inducing_variable=inducing_variable)
    return psi2

def computePsi2(scale, lengthscales, variational_mean, variational_var, inducing_variable):
    # lenthscales # D
    lengthscales_div2 = lengthscales*0.5 # 1xD
    Xmu = variational_mean # NxD
    Xcov = variational_var # NxD
    Z = inducing_variable # MxD
    Zt = np.transpose(Z) # DxM
    Z_u = np.expand_dims(Z, 2) # MxDx1
    all_Zdiffs = Zt - Z_u  # MxDxM
    lengthscales_div2_u = np.expand_dims(lengthscales_div2, 1) # Dx1
    exponent_mahalanobisZ = all_Zdiffs**2/lengthscales_div2_u # MxDxM
    exponential_argZ = -0.125*np.sum(exponent_mahalanobisZ, 1) # MxM
    exponential_valueZ = np.exp(exponential_argZ) # MxM

    all_Zavg = 0.5*(Zt + Z_u)  # MxDxM
    Xmu_u = np.expand_dims(Xmu, 1) # NX1XD
    Xmu_u = np.expand_dims(Xmu_u, 3) # NX1XDx1
    all_XmuZdiffs = Xmu_u - all_Zavg  # NxMxDxM
    Xcov_denom = Xcov+lengthscales_div2 # NxD
    Xcov_denom_u = np.expand_dims(Xcov_denom, 1) # Nx1xD
    Xcov_denom_u = np.expand_dims(Xcov_denom_u, 3) # Nx1xDx1
    exponent_mahalanobisXmuZ = all_XmuZdiffs**2/Xcov_denom_u # NxMxDxM
    exponential_argXmuZ = -0.5*np.sum(exponent_mahalanobisXmuZ, 2) # NxMxM
    exponential_valueXmuZ = np.exp(exponential_argXmuZ) # NxMxM

    psi2_denominator = np.sqrt(Xcov_denom/lengthscales_div2) # NxD
    psi2_denominator_prod = np.prod(psi2_denominator, 1) # N
    psi2_denominator_prod_u = np.expand_dims(psi2_denominator_prod, 1) # Nx1
    psi2_denominator_prod_u = np.expand_dims(psi2_denominator_prod_u, 1) # Nx1x1

    psi2_notSummed = scale**2*exponential_valueZ*exponential_valueXmuZ/psi2_denominator_prod_u # NxMxM
    psi2 = np.sum(psi2_notSummed, 0)
    return psi2
