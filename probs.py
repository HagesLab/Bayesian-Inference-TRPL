from numba import cuda
import numpy as np
import math
def lnP(P, plI, values, mag_grid, bval_cutoff, T_FACTOR):
    for m, mag in enumerate(mag_grid):
        err = plI + mag
        cutoff = np.log10(bval_cutoff)
        err[err < cutoff] = cutoff

        err -= values

        #sig_sq = 1 / (len(plI) - len(X[0])) * np.sum((plI) ** 2, axis=0) # Total error per timestep/observation
        sig_sq = len(values) / T_FACTOR
        #P[:, m] -= np.sum((err)**2 / sig_sq + np.log(np.pi*sig_sq)/2, axis=1)
        P[:, m] -= np.sum((err)**2, axis=1) / sig_sq
        P[:, m] -= np.log(np.pi*sig_sq)/2 * len(values)
    return

@cuda.jit(device=False)
def kernel_lnP(P, plI, values, mag_grid, bval_cutoff, T_FACTOR):
    cutoff = math.log10(bval_cutoff)
    num_observations = len(values)
    num_paramsets = len(plI)
    sig_sq = len(values) / T_FACTOR
    thr = cuda.grid(1)
    for m in range(len(mag_grid)):
        for j in range(thr, num_paramsets, cuda.gridsize(1)):
            
            for i in range(num_observations):
                err = plI[j,i] + mag_grid[m]
                if err < cutoff:
                    err = cutoff

                err -= values[i]

                err = err ** 2

                P[j,m] -= err
            P[j,m] /= sig_sq
            P[j,m] -= math.log(math.pi*sig_sq)/2 * num_observations

    return

if __name__ == "__main__":
    tests = 10
    num_obs = 1000
    num_mags = 10
    num_probs = 1294
    for t in range(tests):
        plI = np.random.uniform(-100, 0, size=(num_probs, num_obs))
        plI2 = np.array(plI)
        try:
            np.testing.assert_equal(plI, plI2)
        except Exception as e:
            print("Uh oh", str(e))
        values = np.random.uniform(-100, 0, size=num_obs)
        mag_grid = np.random.uniform(-100,100, size=num_mags)
        bval_cutoff = 10 ** np.random.uniform(-100,0)
        T_FACTOR = np.random.randint(1,100)
        P1 = np.zeros((num_probs, num_mags))
        P2 = np.zeros((num_probs, num_mags))
        lnP(P1, plI, values, mag_grid, bval_cutoff, T_FACTOR)
        P_dev = cuda.to_device(P2)
        plI_dev = cuda.to_device(plI2)
        kernel_lnP[68, 128](P_dev, plI_dev, values, mag_grid, bval_cutoff, T_FACTOR)
        cuda.synchronize()
        P2 = P_dev.copy_to_host()
        try:
            np.testing.assert_almost_equal(P1, P2)
            print("Test {} passed".format(t))
        except Exception as e:
            print("Test {} failed: {}".format(t, str(e)))

