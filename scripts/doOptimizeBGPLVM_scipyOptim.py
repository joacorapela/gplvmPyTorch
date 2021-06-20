
import sys
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
    variational_var_init = 0.5 # initial value for variational variance
    lr = 1.0
    max_iter = 100000
    tolerance_grad = 1e-7
    tolerance_change = 1e-9
    line_search_fn = "strong_wolfe"
    latents_to_plot = (0, 1)
    fig_title_pattern = "ELBO={:.02f}"
    # data_filename = "/nfs/ghome/live/rapela/dev/research/programs/github/python/GPflow/notebooks/basics/data/three_phase_oil_flow.npz"
    data_filename = "../../../data/DataTrn.txt"
    data_labels_filename = "../../../data/DataTrnLbls.txt"
    lengthscales_fig_filename = "../../../figures/scipyOptim_lengthscales.png"
    latents_fig_filename = "../../../figures/scipyOptim_2latents.png"
    model_save_filename_pattern = "../../../results/{:08d}_scipy_model.pickle"

    file_exists = True
    while file_exists:
        a_random_number = np.random.rand(1)
        model_save_filename = model_save_filename_pattern.format(a_random_number)
        if not os.path.exists(model_save_filename):
            file_exists = False

    # data = np.load(data_filename)
    # Y = torch.tensor(data["Y"], dtype=torch.double)
    # labels = torch.tensor(data["labels"])
    Y = torch.from_numpy(np.genfromtxt(data_filename))
    labels_3cols = torch.from_numpy(np.genfromtxt(data_labels_filename))
    labels = torch.argmax(labels_3cols, axis=1)
    print("Number of points: {:d} and Number of dimensions: {:d}".format(Y.shape[0], Y.shape[1]))
    num_data = Y.shape[0]  # number of data points

    X_mean_init = stats.pca_reduce(X=Y, latent_dim=latent_dim)
#     with open("/nfs/ghome/live/rapela/tmp/X_mean_init.npy", "wb") as f:
#         np.save(f, X_mean_init)
#     import pdb; pdb.set_trace()
#     with open("/nfs/ghome/live/rapela/tmp/X_mean_init.npy", "rb") as f:
#         X_mean_init = torch.from_numpy(np.load(f))

    # X_var_init = torch.ones((num_data, latent_dim), dtype=torch.double)
    X_var_init = variational_var_init*torch.ones((num_data, latent_dim), dtype=torch.double)
    likelihood_var = torch.tensor(0.01)

    torch.manual_seed(1)  # for reproducibility
    # inducing_variable = tf.convert_to_tensor(
    #     np.random.permutation(X_mean_init.numpy())[:num_inducing], dtype=default_float()            
    # )
    # inducing_variable = X_mean_init[torch.randperm(n=X_mean_init.shape[0])][:num_inducing]
    inducing_variable = X_mean_init[:num_inducing]

    lengthscales = torch.tensor([1.0]*latent_dim, dtype=torch.double)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))
    kernel.outputscale = 1.0
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


    def flatten_params(params_iter):
        params_names = []
        flattened = np.array([])
        for param_name, param_value in params_iter:
            params_names.append(param_name)
            flattened = np.concatenate((flattened, param_value.detach().numpy().flatten()))
        return params_names, flattened

    params0_iter = gplvm.named_parameters()
    param_names, x0 = flatten_params(params_iter=params0_iter)

    def set_model_params(flattened_params, param_names=param_names, model=gplvm):
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
            
    def get_model_grad(param_names=param_names, model=gplvm):
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

    def objective(flattened_params, param_names=param_names, model=gplvm):
        set_model_params(flattened_params=flattened_params,
                         param_names=param_names, model=model)
        value = -gplvm.elbo()
        value.backward()
        answer_value = value.detach().numpy()
        answer_grad = get_model_grad(param_names=param_names, model=model)
        print("ELBO: {:f}".format(-value))
        return answer_value, answer_grad


    minimizeOptions = {'ftol': tolerance_change, 'gtol': tolerance_grad, 'maxiter': max_iter}
    optimRes = scipy.optimize.minimize(fun=objective, x0=x0, method='L-BFGS-B',
                                       jac=True, options=minimizeOptions)

    f = open(model_save_filename, "rb")
    pickle.dump(optimRes, f)

    title = fig_title_pattern.format(gplvm.elbo())

    lengthscales = gplvm.kernel.base_kernel.lengthscale.detach().numpy().squeeze()
    plt.bar(np.arange(len(lengthscales))+1, 1./lengthscales)
    plt.xlabel("Latent")
    plt.ylabel("Inverse Lengthscale")
    plt.title(title)
    plt.savefig(lengthscales_fig_filename)

    plt.figure()
    plt.scatter(gplvm.variational_mean.detach().numpy()[:,latents_to_plot[0]], gplvm.variational_mean.detach().numpy()[:,latents_to_plot[1]], c=labels)
    plt.xlabel("Latent {:d}".format(latents_to_plot[0]+1))
    plt.ylabel("Latent {:d}".format(latents_to_plot[1]+1))
    plt.title(title)
    plt.savefig(latents_fig_filename)

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
