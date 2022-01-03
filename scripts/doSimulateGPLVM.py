
import sys
import os.path
import random
import pickle
import configparser
import numpy as np

sys.path.append("../src")
import utilities

def main(argv):
    N = 1000
    D = 12
    Q = 2
    output_scale = 2.0
    lengthscales = [1.0, 3.0]
    likelihood_var = 1e-2
    simResConfig_filename_pattern = \
            "../results/gplvmSimulationMetadata_{:d}.ini"
    simRes_filename_pattern = "../results/gplvmSimulationRes_{:d}.pickle"

    random_prefix_used = True
    while random_prefix_used:
        sim_number = random.randint(0, 10**8)
        simResConfig_filename = \
            simResConfig_filename_pattern.format(sim_number)
        if not os.path.exists(simResConfig_filename):
           random_prefix_used = False
    simRes_filename = simRes_filename_pattern.format(sim_number)

    Y, X = utilities.simulate(N=N, D=D, Q=Q, output_scale=output_scale,
                              lengthscales=lengthscales,
                              likelihood_var=likelihood_var)

    simRes = {"X": X, "Y": Y}
    with open(simRes_filename, "wb") as f: pickle.dump(simRes, f)
    simResConfig = configparser.ConfigParser()
    simResConfig["simulation_params"] = {"N": N, "D": D, "Q": Q,
                                         "output_scale": output_scale,
                                         "lengthscales": lengthscales,
                                         "likelihood_var": likelihood_var}
    simResConfig["simulation_results"] = \
            {"sim_res_filename": simRes_filename}
    with open(simResConfig_filename, "w") as f:
        simResConfig.write(f)

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
