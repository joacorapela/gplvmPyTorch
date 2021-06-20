
import sys
import argparse
import numpy as np
import torch
import gpytorch

sys.path.append("../src")
import models

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("params_filename", help="Parameters filename")
    parser.add_argument("--latent_dim", help="Latent dimensionality", type=int, default=2)
    parser.add_argument("--tol", help="assert tolerance", type=float, default=1e-6)
    parser.add_argument("--data_filename", help="Data filename", default="/nfs/ghome/live/rapela/dev/research/programs/github/python/GPflow/notebooks/basics/data/three_phase_oil_flow.npz")
    args = parser.parse_args()

    params_filename = args.params_filename
    latent_dim = args.latent_dim
    tol = args.tol
    data_filename = args.data_filename

    data = np.load(data_filename)
    Y = torch.tensor(data["Y"], dtype=torch.double)

    params = np.load(params_filename)

    original_elbo = params["bound"]
    psi0 = torch.from_numpy(params["psi0"])
    psi1 = torch.from_numpy(params["psi1"])
    psi2 = torch.from_numpy(params["psi2"])
    X_mean_init = torch.from_numpy(params["variational_mean"])
    X_var_init = torch.from_numpy(params["variational_var"])
    likelihood_var = torch.from_numpy(params["likelihood_variance"])

    inducing_variable = torch.from_numpy(params["inducing_variable"])
    kernel_variance = torch.from_numpy(params["kernel_variance"])
    kernel_lengthscales = torch.from_numpy(params["kernel_lengthscales"])
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))
    kernel.outputscale = kernel_variance
    kernel.base_kernel.lengthscale = kernel_lengthscales

    gplvm = models.BayesianGPLVM(
        data=Y,
        variational_mean=X_mean_init,
        variational_var=X_var_init,
        kernel=kernel,
        inducing_variable=inducing_variable,
        likelihood_var = likelihood_var,
    )
    print("Testing outputscale")
    outputscale = 10.0
    gplvm.kernel.outputscale = outputscale
    assert(gplvm.kernel.outputscale == outputscale)
    assert((gplvm.kernel.raw_outputscale-gplvm.kernel.raw_outputscale_constraint.inverse_transform(gplvm.kernel.raw_outputscale_constraint.transform(gplvm.kernel.raw_outputscale)))<tol)
    print("Test outputscale succeeded")

    print("Testing lengthscale")

    lengthscale = torch.tensor([10.0, 11.0])
    gplvm.kernel.base_kernel.lengthscale = lengthscale
    lengthscale_reconstructed = gplvm.kernel.base_kernel.lengthscale
    for i in range(lengthscale.shape[0]):
        assert((lengthscale[i].item()-lengthscale_reconstructed[0,i].item())<tol)

    raw_lengthscale = gplvm.kernel.base_kernel.raw_lengthscale
    raw_lengthscale_reconstructed = gplvm.kernel.base_kernel.raw_lengthscale_constraint.inverse_transform(gplvm.kernel.base_kernel.raw_lengthscale_constraint.transform(gplvm.kernel.base_kernel.raw_lengthscale))
    for i in range(raw_lengthscale.shape[0]):
        for j in range(raw_lengthscale.shape[1]):
            assert((raw_lengthscale[i,j]-raw_lengthscale_reconstructed[i,j])<tol)

    print("Test lengthscale succeeded")

    print("Testing likelihood_var")
    likelihood_var = torch.tensor([10.0])
    gplvm.likelihood_var = likelihood_var
    assert(gplvm.likelihood_var == likelihood_var)
    assert((gplvm.raw_likelihood_var-gplvm.raw_likelihood_var_constraint.inverse_transform(gplvm.raw_likelihood_var_constraint.transform(gplvm.raw_likelihood_var)))<tol)
    print("Test likelihood_var succeeded")

    print("Testing variational_var")
    variational_var = torch.rand(gplvm.variational_var.shape)
    gplvm.variational_var = variational_var
    for i in range(variational_var.shape[0]):
        for j in range(variational_var.shape[1]):
            assert((variational_var[i,j]-variational_var_reconstructed[i,j])<tol)

    raw_variational_var = gplvm.raw_variational_var
    raw_variational_var_reconstructed = gplvm.raw_variational_var_constraint.inverse_transform(gplvm.raw_variational_var_constraint.transform(gplvm.raw_variational_var))
    for i in range(raw_variational_var.shape[0]):
        for j in range(raw_variational_var.shape[1]):
            assert((raw_variational_var[i,j]-raw_variational_var_reconstructed[i,j])<tol)
    print("Test variational_var succeeded")

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
