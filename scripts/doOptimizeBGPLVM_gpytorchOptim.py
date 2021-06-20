
import sys
import numpy as np
import torch
import gpytorch

import matplotlib.pyplot as plt

sys.path.append("../src")

import models
import stats
import utilities

def main(argv):
    nudget = 1e-6
    latent_dim = 10  # number of latent dimensions
    num_inducing = 50  # number of inducing pts
    variational_var_init = 0.5 # initial value for variational variance
    lr = 1.0
    max_iter = 1000
    line_search_fn = "strong_wolfe"
    latents_to_plot = (0, 1)
    # data_filename = "/nfs/ghome/live/rapela/dev/research/programs/github/python/GPflow/notebooks/basics/data/three_phase_oil_flow.npz"
    data_filename = "../../data/DataTrn.txt"
    data_labels_filename = "../../data/DataTrnLbls.txt"
    lengthscales_fig_filename = "../../figures/gpytorchOptim_lengthscales.png"
    latents_fig_filename = "../../figures/gpytorchOptim_2latents.png"

    # data = np.load(data_filename)
    # Y = torch.tensor(data["Y"], dtype=torch.double)
    Y = torch.from_numpy(np.genfromtxt(data_filename))
    labels_3cols = torch.from_numpy(np.genfromtxt(data_labels_filename))
    labels = torch.argmax(labels_3cols, axis=1)
    # import pdb; pdb.set_trace()
    print("Number of points: {:d} and Number of dimensions: {:d}".format(Y.shape[0], Y.shape[1]))
    num_data = Y.shape[0]  # number of data points

    X_mean_init = stats.pca_reduce(X=Y, latent_dim=latent_dim)
#     with open("/nfs/ghome/live/rapela/tmp/X_mean_init.npy", "wb") as f:
#         np.save(f, X_mean_init)
#     import pdb; pdb.set_trace()
#     with open("/nfs/ghome/live/rapela/tmp/X_mean_init.npy", "rb") as f:
#         X_mean_init = torch.from_numpy(np.load(f))

    X_var_init = variational_var_init*torch.ones((num_data, latent_dim), dtype=torch.double)
    likelihood_var = torch.tensor(0.01, dtype=torch.double)

    torch.manual_seed(1)  # for reproducibility
    # inducing_variable = tf.convert_to_tensor(
    #     np.random.permutation(X_mean_init.numpy())[:num_inducing], dtype=default_float()            
    # )
    # inducing_variable = X_mean_init[torch.randperm(n=X_mean_init.shape[0])][:num_inducing]
    inducing_variable = X_mean_init[:num_inducing]

    lengthscales = torch.tensor([1.0]*latent_dim, dtype=torch.double)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))
    kernel.outputscale = torch.tensor([1.0], dtype=torch.double)
    kernel.base_kernel.lengthscale = lengthscales
    gplvm = models.BayesianGPLVM(
        data=Y,
        variational_mean=X_mean_init,
        variational_var=X_var_init,
        kernel=kernel,
        inducing_variable=inducing_variable,
        likelihood_var = likelihood_var,
        nudget = nudget,
    )
    gplvm = gplvm.double()

#     named_params = gplvm.named_parameters()
#     for param_name, param_value in named_params:
#         print(param_name, param_value.dtype)
#         import pdb; pdb.set_trace()
    x = gplvm.parameters()
    optimizer = torch.optim.LBFGS(x, lr=lr, max_iter=max_iter,
                                  line_search_fn=line_search_fn)

    def closure():
        # details on this closure at http://sagecal.sourceforge.net/pytorch/index.html
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        cur_eval = -gplvm.elbo()
        if cur_eval.requires_grad:
            cur_eval.backward(retain_graph=True)
        print("ELBO: {:f}".format(-cur_eval))
        return cur_eval

    optimizer.step(closure)

    lengthscales = gplvm.kernel.base_kernel.lengthscale.detach().numpy().squeeze()
    plt.bar(np.arange(len(lengthscales))+1, 1./lengthscales)
    plt.xlabel("Latent")
    plt.ylabel("Inverse Lengthscale")
    plt.savefig(lengthscales_fig_filename)

    plt.figure()
    plt.scatter(gplvm.variational_mean.detach().numpy()[:,latents_to_plot[0]], gplvm.variational_mean.detach().numpy()[:,latents_to_plot[1]], c=labels)
    plt.xlabel("Latent {:d}".format(latents_to_plot[0]+1))
    plt.ylabel("Latent {:d}".format(latents_to_plot[1]+1))
    plt.savefig(latents_fig_filename)
    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
