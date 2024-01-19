import numpy as np
import datetime as dt

import scipy.signal as ss
import matplotlib.pyplot as plt

import pyxu.operator as pxop
import pyxu.opt.stop as pxos

import pyfwl

N = 100
k = 10
L = 3 * k
psnrdb = 20

sigma = 2
Nl = N + 6 * int(sigma)

if __name__ == "__main__":
    seed = 42
    rng = np.random.default_rng(seed)

    convop = 2 * pxop.Gaussian((Nl,), sigma=sigma)
    samples = rng.choice(Nl, size=L, replace=False)
    S = pxop.SubSample((Nl,), samples)

    fOp = S * convop
    fOp.lipschitz = fOp.estimate_lipschitz(method='svd', tol=1.e-2)
    # %%
    supp = rng.choice(N, size=k, replace=False) + 3 * int(sigma)
    x = np.zeros(N + 6 * int(sigma))
    x[supp] = rng.uniform(2, 6, size=k)

    g = convop(x)
    noiselessy = fOp(x)
    std = np.max(np.abs(noiselessy)) * 10 ** (-psnrdb / 20)
    noise = rng.normal(0, std, size=L)

    y = noiselessy + noise

    plt.figure()
    m, s, b = plt.stem(np.where(x > 0)[0], x[x > 0], label='x')
    b._visible = False
    plt.scatter(np.arange(Nl), g, label='g', marker='x', color='r')
    plt.scatter(samples, y, label='samples', marker='o', color='none', edgecolors='g', s=200)
    plt.legend()
    plt.show()

    ## Dual certificate
    d = fOp.adjoint(y)

    plt.figure()
    plt.scatter(np.arange(Nl), d, marker='.', color='r', label="Dirty image")
    plt.scatter(np.arange(Nl), g, marker='.', color='orange', label="Source")
    _, _, b = plt.stem(np.where(x > 0)[0], x[x > 0], label='x')
    b._visible = False
    plt.legend()
    plt.show()

    # Solve with PFW
    lambda_factor = 0.1
    lambda_ = lambda_factor * np.max(np.abs(d))

    eps = 1e-2
    min_iterations = 10

    # plot dual certificate at beginning
    plt.figure()
    plt.scatter(np.arange(Nl), d / lambda_, marker='.', color='r', label="Initial dual certificate")
    plt.hlines([-1, 1], 0, Nl, color='orange', label="Threshold y=1")
    plt.legend()
    plt.show()

    stop_crit = pxos.RelError(
        eps=eps,
        var="objective_func",
        f=None,
        norm=2,
        satisfy_all=True,
    )

    # Minimum number of iterations
    min_iter = pxos.MaxIter(n=min_iterations)
    # Maximum runtime
    tmax = 15
    # track DCV
    track_dcv = pxos.AbsError(eps=1e-10, var="dcv", f=None, norm=2, satisfy_all=True)
    stop_crit = (min_iter & stop_crit) | pxos.MaxDuration(t=dt.timedelta(seconds=tmax)) | track_dcv

    ## Regular PFW
    pfw = pyfwl.PFWLasso(y, fOp, lambda_,
                         ms_threshold=0.9,
                         final_correction_prec=eps)

    pfw.fit(stop_crit=stop_crit, verbose=50,
            precision_rule=lambda k: 10 ** (-k / 10))

    solpfw = pfw.stats()[0]['x']

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(np.arange(Nl), g, marker='.', color='r', label="Source")
    plt.scatter(np.arange(Nl), convop(solpfw), marker='.', color='orange', label="Solution")
    plt.legend()

    plt.subplot(122)
    plt.scatter(np.arange(Nl), x, marker='.', color='r', label="Sparse source")
    plt.scatter(np.arange(Nl), solpfw, marker='.', color='orange', label="Solution")
    plt.legend()
    plt.show()

    pfw.diagnostics()

    ## Solve with LocalMaxPFW
    from pyfwl import LocalMaxPFW

    lmpfw = LocalMaxPFW(y, fOp, lambda_, final_correction_prec=eps)

    lmpfw.fit(stop_crit=stop_crit, verbose=50,
              precision_rule=lambda k: 10 ** (-k / 10))

    lmpfw.diagnostics()

    sollm = lmpfw.stats()[0]['x']

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(np.arange(Nl), g, marker='.', color='r', label="Source")
    plt.scatter(np.arange(Nl), convop(sollm), marker='.', color='orange', label="Solution")
    plt.legend()

    plt.subplot(122)
    plt.scatter(np.arange(Nl), x, marker='.', color='r', label="Sparse source")
    plt.scatter(np.arange(Nl), sollm, marker='.', color='orange', label="Solution")
    plt.legend()
    plt.show()

    # Compare the solution of PFW and LMPFW
    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(Nl), x, marker='.', color='b', label="Sparse source")
    plt.scatter(np.arange(Nl), solpfw, marker='+', color='r', label="PFW", alpha=.5)
    plt.scatter(np.arange(Nl), sollm, marker='x', color='orange', label="Local Max", alpha=.5)
    plt.legend()
    plt.show()

    ## Solve with PGD
    from pyxu.opt.solver import PGD

    # Explicit definition of the objective function for APGD
    data_fid = 0.5 * pxop.SquaredL2Norm(dim=fOp.shape[0]).argshift(-y) * fOp
    regul = lambda_ * pxop.L1Norm(N)

    print("Solving with APGD: ...")
    pgd = PGD(data_fid, regul, verbosity=50)
    pgd.fit(
        x0=np.zeros(Nl, dtype="float64"),
        stop_crit=(min_iter & pgd.default_stop_crit()) | pxos.MaxDuration(t=dt.timedelta(seconds=tmax)),
        track_objective=True,
        tau=1 / fOp.lipschitz ** 2,
    )
    sol_apgd = pgd.stats()[0]["x"]

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(np.arange(Nl), g, marker='.', color='r', label="Source")
    plt.scatter(np.arange(Nl), convop(sol_apgd), marker='.', color='orange', label="Solution")
    plt.legend()

    plt.subplot(122)
    plt.scatter(np.arange(Nl), x, marker='.', color='r', label="Sparse source")
    plt.scatter(np.arange(Nl), sol_apgd, marker='.', color='orange', label="Solution")
    plt.legend()
    plt.show()


    # %%
    # 2D
    import skimage.feature as skf

    N = 100
    a, b = 15, 20
    # create a 2d sinusoidal image of size N x N
    xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    z = np.sin(2 * np.pi * (xx / a + yy / b)) * np.exp(-0.001 * ((xx - 50) ** 2 + (yy - 50) ** 2))

    plt.figure()
    plt.imshow(z, origin='lower', cmap='viridis')
    for (x, y) in skf.peak_local_max(z, threshold_abs=.4, min_distance=2, p_norm=np.inf):
        plt.scatter(x, y, marker='x', color='r')
    plt.colorbar()
    plt.show()


