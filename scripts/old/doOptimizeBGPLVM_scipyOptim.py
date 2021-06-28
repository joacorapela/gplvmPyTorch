
import sys
import os.path
import pickle
import numpy as np
import torch
import gpytorch
import scipy.optimize

import matplotlib.pyplot as plt

sys.path.append("../src")

import models
import stats
import utilities

def main(argv):
    nudget = 1e-6
    latent_dim = 10  # number of latent dimensions
    num_inducing = 50  # number of inducing pts
    variational_var0 = 0.5 # initial value for variational variance
    kernel_output_scale0 = 1.0
    likelihood_var0 = 0.01
    lengthscale0 = 1.0
    lr = 1.0
    max_iter = 100000
    max_fun = 4*max_iter
    tolerance_grad = 1e-7
    tolerance_change = 1e-9
    line_search_fn = "strong_wolfe"
    latents_to_plot = (1, 2)
    fig_title_pattern = "ELBO={:.02f}"
    data_filename = "../../data/DataTrn.txt"
    data_labels_filename = "../../data/DataTrnLbls.txt"
    lengthscales_fig_filename_pattern = "../../figures/{:08d}_scipyOptim_lengthscales.png"
    latents_fig_filename_pattern = "../../figures/{:08d}_scipyOptim_2latents.png"
    model_save_filename_pattern = "../../results/{:08d}_scipy_model.npz"

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

    X_var_init = variational_var0*torch.ones((num_data, latent_dim), dtype=torch.double)
    likelihood_var = torch.tensor(likelihood_var0)

    torch.manual_seed(1)  # for reproducibility
    inducing_variable = X_mean_init[torch.randperm(n=X_mean_init.shape[0])][:num_inducing]

    lengthscales = torch.tensor([lengthscale0]*latent_dim, dtype=torch.double)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))
    kernel.outputscale = kernel_output_scale0
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

    params0_iter = gplvm.named_parameters()
    param_names, x0 = utilities.flatten_params(params_iter=params0_iter)

    def objective(flattened_params, param_names=param_names, model=gplvm):
        utilities.set_model_params(flattened_params=flattened_params, param_names=param_names, model=model)
        value = -gplvm.elbo()
        value.backward()
        answer_value = value.detach().numpy()
        answer_grad = utilities.get_model_grad(param_names=param_names, model=model)
        print("ELBO: {:f}".format(-value))
        return answer_value, answer_grad

    minimizeOptions = {'ftol': tolerance_change, 'gtol': tolerance_grad,
                       'maxiter': max_iter, 'maxfun': max_fun}
    optimRes = scipy.optimize.minimize(fun=objective, x0=x0, method='L-BFGS-B',
                                       jac=True, options=minimizeOptions)

    # Finally, save and plot results
    model_info = dict(Y=Y, variational_mean=gplvm.variational_mean.detach().numpy(), variational_var=gplvm.variational_var.detach().numpy(), kernel_outputscale=gplvm.kernel.outputscale.detach().numpy(), kernel_lengthscale=gplvm.kernel.base_kernel.lengthscale.detach().numpy(), inducing_variable=gplvm.inducing_variable.detach().numpy(), nudget=gplvm.nudget)
    with open(model_save_filename, "wb") as f: np.savez(f, **model_info)

    title = fig_title_pattern.format(gplvm.elbo())

    lengthscales = gplvm.kernel.base_kernel.lengthscale.detach().numpy().squeeze()
    plt.bar(np.arange(len(lengthscales))+1, 1./lengthscales)
    plt.xlabel("Latent")
    plt.ylabel("Inverse Lengthscale")
    plt.title(title)
    lengthscales_fig_filename = lengthscales_fig_filename_pattern.format(a_random_number)
    plt.savefig(lengthscales_fig_filename)

    variational_means = gplvm.variational_mean.detach().numpy()[:,latents_to_plot]

    plt.figure()
    plt.gca().invert_yaxis()
    plt.scatter(variational_means[:,0], variational_means[:,1], c=labels)
    plt.xlabel("Latent {:d}".format(latents_to_plot[0]+1))
    plt.ylabel("Latent {:d}".format(latents_to_plot[1]+1))
    plt.title(title)
    latents_fig_filename = latents_fig_filename_pattern.format(a_random_number)
    plt.savefig(latents_fig_filename)

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
