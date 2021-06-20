
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
    parser.add_argument("--use_gpflow_psi012", help="Use psi0, psi1 and psi2 calculated in GPflow", action="store_true")
    parser.add_argument("--latent_dim", help="Latent dimensionality", type=int, default=2)
    parser.add_argument("--num_inducing", help="Number of inducing points", type=int, default=20)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1.0)
    parser.add_argument("--max_iter", help="Maximum number of iterations", type=int, default=100)
    parser.add_argument("--line_search_fn", help="Name of line search function routine", default="strong_wolfe")
    parser.add_argument("--data_filename", help="Data filename", default="/nfs/ghome/live/rapela/dev/research/programs/github/python/GPflow/notebooks/basics/data/three_phase_oil_flow.npz")
    args = parser.parse_args()

    use_gpflow_psi012 = args.use_gpflow_psi012
    latent_dim = args.latent_dim
    num_inducing = args.num_inducing
    lr = args.lr
    max_iter = args.max_iter
    line_search_fn = args.line_search_fn
    params_filename = args.params_filename
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
    if use_gpflow_psi012:
        calculated_elbo = gplvm.elbo(psi0=psi0, psi1=psi1, psi2=psi2)
    else:
        calculated_elbo = gplvm.elbo()
    print("original={:f} and calculated={:f} ELBOs".format(original_elbo, calculated_elbo))

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
