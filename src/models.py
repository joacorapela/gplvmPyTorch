
import math
import warnings
import torch
import torch.nn as nn
import gpytorch
import utilities

class BayesianGPLVM(gpytorch.Module):

    def __init__(self, data, variational_mean, variational_var, kernel,
                 inducing_variable, likelihood_var,
                 variational_var_constraint=None,
                 likelihood_var_constraint=None,
                 prior_mean=None, prior_var=None):
        super(BayesianGPLVM, self).__init__()

        self.data = data
        self.variational_mean = nn.Parameter(variational_mean)

        # variational_var
        if variational_var_constraint is None:
            variational_var_constraint = gpytorch.constraints.Positive()
        self.variational_var_constraint = variational_var_constraint
        self.register_parameter(
            name="raw_variational_var",
            parameter=torch.nn.Parameter(variational_var_constraint.inverse_transform(variational_var))
        )
        self.register_constraint("raw_variational_var", variational_var_constraint)

        self.kernel = kernel
        self.inducing_variable = nn.Parameter(inducing_variable)

        # likelihood_var
        if likelihood_var_constraint is None:
            likelihood_var_constraint = gpytorch.constraints.Positive()
        self.likelihood_var_constraint = likelihood_var_constraint
        self.register_parameter(
            name="raw_likelihood_var",
            parameter=torch.nn.Parameter(likelihood_var_constraint.inverse_transform(likelihood_var))
        )
        self.register_constraint("raw_likelihood_var", likelihood_var_constraint)

        # deal with parameters for the prior mean var of X
        num_obs, dim_latents = variational_mean.shape
        if prior_mean is None:
            prior_mean = torch.zeros((num_obs, dim_latents), dtype=torch.double)
        if prior_var is None:
            prior_var = torch.ones((num_obs, dim_latents), dtype=torch.double)
        self.prior_mean = prior_mean
        self.prior_var = prior_var

    @property
    def variational_var(self):
        value = self.variational_var_constraint.transform(self.raw_variational_var)
        return value

    @variational_var.setter
    def variational_var(self, value):
        self._set_variational_var(value)

    def _set_variational_var(self, value):
        self.initialize(raw_variational_var=self.variational_var_constraint.inverse_transform(value))

    @property
    def likelihood_var(self):
        value = self.likelihood_var_constraint.transform(self.raw_likelihood_var)
        return value

    @likelihood_var.setter
    def likelihood_var(self, value):
        self._set_likelihood_var(value)

    def _set_likelihood_var(self, value):
        self.initialize(raw_likelihood_var=self.raw_likelihood_var_constraint.inverse_transform(value))

    def elbo(self, psi0=None, psi1=None, psi2=None, write_debug=True):
        # import pdb; pdb.set_trace()
        Y_data = self.data

        N = Y_data.shape[0]
        num_inducing = self.inducing_variable.shape[0]
        if psi0 is None:
            psi0 = N*self.kernel.outputscale
        if psi1 is None:
            psi1 = utilities.computePsi1(scale=self.kernel.outputscale,
                                         lengthscales=self.kernel.base_kernel.lengthscale,
                                         variational_mean=self.variational_mean,
                                         variational_var=self.variational_var,
                                         inducing_variable=self.inducing_variable)
        else:
            my_psi1 = utilities.computePsi1(scale=self.kernel.outputscale,
                                         lengthscales=self.kernel.base_kernel.lengthscale,
                                         variational_mean=self.variational_mean,
                                         variational_var=self.variational_var,
                                         inducing_variable=self.inducing_variable)
        if psi2 is None:
            psi2 = utilities.computePsi2(scale=self.kernel.outputscale,
                                         lengthscales=self.kernel.base_kernel.lengthscale,
                                         variational_mean=self.variational_mean,
                                         variational_var=self.variational_var,
                                         inducing_variable=self.inducing_variable)
        else:
            my_psi2 = utilities.computePsi2(scale=self.kernel.outputscale,
                                         lengthscales=self.kernel.base_kernel.lengthscale,
                                         variational_mean=self.variational_mean,
                                         variational_var=self.variational_var,
                                         inducing_variable=self.inducing_variable)
        cov_uu = self.kernel(self.inducing_variable).evaluate()
        # begin debug
        e_values, e_vectors = torch.eig(cov_uu)
        if e_values.min()<0:
            # import pdb; pdb.set_trace()
            warnings.warn("Found negative eigenvalues of cov_uu")
        # end debug
        L = torch.cholesky(cov_uu)
        sigma2 = self.likelihood_var

        # Compute intermediate matrices
        A = torch.triangular_solve(torch.transpose(psi1, 0, 1), L, upper=False).solution
        tmp = torch.triangular_solve(psi2, L, upper=False).solution
        AAT = torch.triangular_solve(torch.transpose(tmp, 0, 1), L, upper=False).solution / sigma2
        B = AAT + torch.eye(num_inducing, dtype=torch.double)
        # begin debug
        # e_values, e_vectors = torch.eig(B)
        # import pdb; pdb.set_trace()
        # end debug
        LB = torch.cholesky(B)
        log_det_B = 2.0 * torch.sum(torch.log(torch.diag(LB)))
        c = torch.triangular_solve(torch.matmul(A, Y_data), LB, upper=False).solution / sigma2

        # KL[q(x) || p(x)]
        NQ = self.prior_mean.numel()
        D = Y_data.shape[1]
        KL = -0.5 * torch.sum(torch.log(self.variational_var))
        KL += 0.5 * torch.sum(torch.log(self.prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * torch.sum(
            (torch.square(self.variational_mean - self.prior_mean) + self.variational_var) / self.prior_var
        )

        # compute log marginal bound
        ND = Y_data.numel()
        bound = -0.5 * ND * torch.log(2 * math.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * torch.sum(torch.square(Y_data)) / sigma2
        bound += 0.5 * torch.sum(torch.square(c))
        bound += -0.5 * D * (torch.sum(psi0) / sigma2 - torch.sum(torch.diag(AAT)))
        bound -= KL
        # begin debug
#         if write_debug:
#             print("Bound=", bound)
#             with open("/tmp/pytorch_params.txt", "a") as f:
#                 f.write("Bound={:f}\n".format(bound))
#                 f.write("variational_mean\n")
#                 f.write(str(self.variational_mean))
#                 f.write("\nvariational_var\n")
#                 f.write(str(self.variational_var))
#                 f.write("\ninducing_variable\n")
#                 f.write(str(self.inducing_variable))
#                 f.write("\nlikelihood_var\n")
#                 f.write(str(self.likelihood_var))
#                 f.write("\nkernel outputscale\n")
#                 f.write(str(self.kernel.outputscale))
#                 f.write("\nkernel lengthscale\n")
#                 f.write(str(self.kernel.base_kernel.lengthscale))
#                 f.write("\n################################################################################\n")
            # import pdb; pdb.set_trace()
            # print("Bound=", bound)
        # end debug
        # import pdb; pdb.set_trace()
        return bound

