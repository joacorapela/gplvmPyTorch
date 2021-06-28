
import sys
import os.path
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
    kernel_output_scale0 = 1.0
    likelihood_var0 = 0.01
    lengthscale0 = 1.0
    lr = 1.0
    max_iter = 10000
    line_search_fn = "strong_wolfe"
    fig_title_pattern = "ELBO={:.02f}"
    data_filename = "../data/DataTrn.txt"
    data_labels_filename = "../data/DataTrnLbls.txt"
    elbo_fig_filename_pattern = "../figures/{:08d}_pytorchOptim_elbo.png"
    lengthscales_fig_filename_pattern = "../figures/{:08d}_pytorchOptim_lengthscales.png"
    latents_fig_filename_pattern = "../figures/{:08d}_pytorchOptim_2latents.png"
    model_save_filename_pattern = "../results/{:08d}_pytorch_model.npz"

    file_exists = True
    while file_exists:
        a_random_number = np.random.randint(0, 1e8)
        model_save_filename = model_save_filename_pattern.format(a_random_number)
        if not os.path.exists(model_save_filename):
            file_exists = False

    Y = torch.from_numpy(np.genfromtxt(data_filename))
    labels_3cols = torch.from_numpy(np.genfromtxt(data_labels_filename))
    labels = torch.argmax(labels_3cols, axis=1)
    print("Number of points: {:d} and Number of dimensions: {:d}".format(Y.shape[0], Y.shape[1]))
    num_data = Y.shape[0]  # number of data points

    X_mean_init = stats.pca_reduce(X=Y, latent_dim=latent_dim)

    X_var_init = variational_var_init*torch.ones((num_data, latent_dim), dtype=torch.double)
    likelihood_var = torch.tensor(likelihood_var0, dtype=torch.double)

    torch.manual_seed(1)  # for reproducibility
    inducing_variable = X_mean_init[torch.randperm(n=X_mean_init.shape[0])][:num_inducing]

    lengthscales = torch.tensor([lengthscale0]*latent_dim, dtype=torch.double)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))
    kernel.outputscale = torch.tensor([kernel_output_scale0], dtype=torch.double)
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

    x = gplvm.parameters()
    optimizer = torch.optim.LBFGS(x, lr=lr, max_iter=max_iter,
                                  line_search_fn=line_search_fn)
    elbo_list = np.empty(10*max_iter, dtype=np.double)
    elbo_list[:] = np.NaN
    iter_nro = 0
    def closure():
        nonlocal iter_nro
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        cur_eval = -gplvm.elbo()
        if cur_eval.requires_grad:
            cur_eval.backward(retain_graph=True)
        print("ELBO: {:f}, iter: {:d}".format(-cur_eval, iter_nro))
        elbo_list[iter_nro] = -cur_eval
        iter_nro += 1
        return cur_eval

    aux = optimizer.step(closure)

    title = fig_title_pattern.format(gplvm.elbo())

    elbo_list[elbo_list<=0] = np.NaN
    plt.figure()
    plt.plot(np.log(elbo_list))
    plt.xlabel("Iteration")
    plt.ylabel("log(Evidence Lower Bound)")
    plt.title(title)
    elbo_fig_filename = elbo_fig_filename_pattern.format(a_random_number)
    plt.savefig(elbo_fig_filename)

    lengthscales = gplvm.kernel.base_kernel.lengthscale.detach().numpy().squeeze()
    plt.figure()
    plt.bar(np.arange(len(lengthscales))+1, 1./lengthscales)
    plt.xlabel("Latent")
    plt.ylabel("Inverse Lengthscale")
    plt.title(title)
    lengthscales_fig_filename = lengthscales_fig_filename_pattern.format(a_random_number)
    plt.savefig(lengthscales_fig_filename)

    latents_to_plot = np.argsort(lengthscales)[:2]
    variational_means = gplvm.variational_mean.detach().numpy()[:,latents_to_plot]

    plt.figure()
    plt.scatter(variational_means[:,1], variational_means[:,0], c=labels)
    plt.xlabel("Latent {:d}".format(latents_to_plot[1]+1))
    plt.ylabel("Latent {:d}".format(latents_to_plot[0]+1))
    plt.title(title)
    latents_fig_filename = latents_fig_filename_pattern.format(a_random_number)
    plt.savefig(latents_fig_filename)

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
