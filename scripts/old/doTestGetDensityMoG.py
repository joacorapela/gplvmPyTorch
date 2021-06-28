import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../src")
import stats

def main(argv):
    x = np.arange(0, 100, .1)
    y = np.arange(0, 50, .1)
    means = np.array([[25.,10.], [50.,20.], [75.,30.]], dtype=np.double)
    stds = np.array([[1.,5.], [5.,5.], [10.,3.]], dtype=np.double)

    mog_density = stats.get_density_mixture_of_2DIndependentGaussians(x=x, y=y,
                                                                      means=means,
                                                                      variances=stds**2)
    plt.imshow(mog_density, extent=[x.min(), x.max(), y.max(), y.min()], cmap="gray")
    plt.gca().invert_yaxis()
    plt.savefig("../../figures/test_mog_density.png")
    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
