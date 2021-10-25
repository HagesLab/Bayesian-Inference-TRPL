from numba import cuda, float64
import numpy as np
import math
import time
def lnP(P, plI, values, mag_grid, bval_cutoff, T_FACTOR):
    for m, mag in enumerate(mag_grid):
        err = plI + mag
        cutoff = np.log10(bval_cutoff)
        err[err < cutoff] = cutoff

        err -= values

        #sig_sq = 1 / (len(plI) - len(X[0])) * np.sum((plI) ** 2, axis=0) # Total error per timestep/observation
        sig_sq = T_FACTOR
        #P[:, m] -= np.sum((err)**2 / sig_sq + np.log(np.pi*sig_sq)/2, axis=1)
        P[:, m] -= np.sum((err)**2, axis=1) / sig_sq
        P[:, m] -= np.log(np.pi*sig_sq)/2 * len(values)
    return

@cuda.jit(device=False)
def kernel_lnP(P, plI, values, uncertainty, mag_grid, DEBUG_w):
    #cutoff = math.log10(bval_cutoff)
    err_arr = cuda.shared.array(shape=(shared_array_size,), dtype=float64)
    thr = cuda.grid(1)
    thr2 = cuda.threadIdx.x
    #for ti in range(len(T_FACTORS)):
    #    tf = T_FACTORS[ti]
    for j in range(thr, len(plI), cuda.gridsize(1)):
        err_arr[thr2] = 0

        for i in range(len(values)):
            err = plI[j,i] + mag_grid[j]
            #if err < cutoff:
            #    err = cutoff

            err -= values[i]

            err = err ** 2
            #err /= (2 * uncertainty[i] ** 2)
            err_arr[thr2] += err * DEBUG_w[i]


        P[j] -= err_arr[thr2]
        #P[j] /= tf
        #P[j] -= math.log(math.pi*sig_sq)/2 * num_observations

    return

def prob(P, plI, values, uncertainty, mag_grid, TPB, BPG):
    global shared_array_size
    clock0 = time.time()
    shared_array_size = int(TPB)
    weight = np.ones_like(values)
    #weight[40880:83240] += 1
    #weight[100990:120500] += 1
    v_dev = cuda.to_device(values)
    u_dev = cuda.to_device(uncertainty)
    m_dev = cuda.to_device(mag_grid)
    plI_dev = cuda.to_device(plI)
    P_dev = cuda.to_device(np.zeros_like(P))
    DEBUG_w_dev = cuda.to_device(weight)
    kernel_lnP[BPG, TPB](P_dev, plI_dev, v_dev, u_dev, m_dev, DEBUG_w_dev)
    cuda.synchronize()
    P += P_dev.copy_to_host()

    return time.time() - clock0

@cuda.jit(device=False)
def log_kernel(plI, MIN, TPB, BPG):
    blk = cuda.blockIdx.x
    thr = cuda.threadIdx.x

    for i in range(blk, len(plI), BPG):
        for j in range(thr, len(plI[0]), TPB):
            if plI[i,j] < MIN:
                plI[i,j] = MIN

            plI[i,j] = math.log10(plI[i,j])


def fastlog(plI, MIN, TPB, BPG):
    clock0 = time.time()
    plI_dev = cuda.to_device(plI)
    log_kernel[BPG, TPB](plI_dev, MIN, TPB, BPG)
    cuda.synchronize()
    plI[:] = plI_dev.copy_to_host()

    return time.time() - clock0

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

