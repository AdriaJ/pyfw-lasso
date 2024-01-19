import numpy as np
import datetime as dt
import os

import scipy.signal as ss
import matplotlib.pyplot as plt

import pyxu.operator as pxop
import pyxu.opt.stop as pxos

import pyfwl

# Try the 2D sparse reconstructoin problem
# Create a sparse image of size N=100 times 100, then convolve it with a 2D isotropic Gaussian kernel
# and subsample it.
# Then try to reconstruct the original image with PFW and LocalMaxPFW

N = 100
k = 10
L = 5 * k
psnrdb = 50

sigma = 20
Nl = N + 6 * int(sigma)
imshape = (Nl, Nl)

lambda_factor = 0.1

eps = 1e-4
min_iterations = 10
tmax = 30
dcv_tol = 5e-2

if __name__ == "__main__":
    seed = 42
    rng = np.random.default_rng(seed)

    x = np.zeros((N, N))
    supp = rng.choice(N ** 2, size=k, replace=False)
    x.ravel()[supp] = rng.uniform(2, 6, size=k)
    x = np.pad(x, 3 * int(sigma), mode='constant', constant_values=0)

    # plt.figure()
    # plt.imshow(x, origin='lower', cmap='viridis')
    # plt.show()

    convop = 2 * pxop.Gaussian(imshape, sigma=sigma)
    samples = rng.choice(Nl ** 2, size=L, replace=False)
    S = pxop.SubSample((Nl * Nl,), samples)
    fOp = S * convop
    fOp.lipschitz = fOp.estimate_lipschitz(method='svd', tol=1.e-2)

    g = convop(x.ravel())

    # plt.figure()
    # plt.imshow(g.reshape(imshape), origin='lower', cmap='viridis')
    # plt.show()

    noiselessy = fOp(x.ravel())
    std = np.max(np.abs(noiselessy)) * 10 ** (-psnrdb / 20)
    noise = rng.normal(0, std, size=L)

    y = noiselessy + noise

    ## Dirty image
    d = fOp.adjoint(y)

    # Solve with PFW
    lambda_ = lambda_factor * np.max(np.abs(d))

    # plt.figure()
    # plt.imshow(d.reshape(imshape) / lambda_, origin='lower', cmap='viridis')
    # plt.colorbar()
    # plt.contour(d.reshape(imshape) / lambda_, levels=[1], colors='r')
    # plt.title("Dirty image")
    # plt.show()

    ## Stop criteria
    stop_crit = pxos.RelError(
        eps=eps,
        var="objective_func",
        f=None,
        norm=2,
        satisfy_all=True,
    )
    dcv_stop = pyfwl.dcvStoppingCrit(dcv_tol)

    # Minimum number of iterations
    min_iter = pxos.MaxIter(n=min_iterations)

    # track DCV
    track_dcv = pxos.AbsError(eps=1e-10, var="dcv", f=None, norm=2, satisfy_all=True)
    stop_crit = (min_iter & dcv_stop) | pxos.MaxDuration(t=dt.timedelta(seconds=tmax)) | track_dcv

    ## Regular PFW
    pfw = pyfwl.PFWLasso(y, fOp, lambda_,
                         ms_threshold=0.9,
                         init_correction_prec=1e-2,
                         final_correction_prec=1e-6,
                         show_progress=False,
                         record_certificate=1)

    pfw.fit(stop_crit=stop_crit, verbose=50,
            precision_rule=lambda _: 10 ** (-_ / 3))

    solpfw = pfw.stats()[0]['x']

    ## LocalMaxPFW
    from pyfwl import LocalMaxPFW2D

    lmpfw = LocalMaxPFW2D(imshape, y, fOp, lambda_,
                          init_correction_prec=1e-2,
                          final_correction_prec=1e-6,
                          show_progress=False,
                          record_certificate=1)

    lmpfw.fit(stop_crit=stop_crit, verbose=50,
              precision_rule=lambda _: 10 ** (-_ / 3))
    sollm = lmpfw.stats()[0]['x']

    ## -----------------------------------------------

    ## Analysis of the results

    pfw.diagnostics()
    lmpfw.diagnostics()

    vmaxsparse = max(x.max(), sollm.max(), solpfw.max())
    vmaxconv = max(g.max(), convop(solpfw).max(), convop(sollm).max())
    import matplotlib.colors as mplc
    normsparse = mplc.PowerNorm(gamma=1 / 5, vmin=0, vmax=vmaxsparse)
    normconv = mplc.PowerNorm(gamma=1 / 5, vmin=0, vmax=vmaxconv)

    fig = plt.figure(figsize=(11, 11))
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    ax = axes.ravel()[0]
    ax.imshow(x, origin='lower', cmap='Greys', norm=normsparse)
    ax.set_title('Sparse source')
    ax = axes.ravel()[1]
    ax.imshow(g.reshape(imshape), origin='lower', cmap='viridis', norm=normconv)
    ax.scatter(np.unravel_index(samples, imshape)[1], np.unravel_index(samples, imshape)[0], marker='x', color='r')
    ax.set_title('Convolved source')
    ax = axes.ravel()[2]
    ax.imshow(solpfw.reshape(imshape), origin='lower', cmap='Greys', norm=normsparse)
    ax.set_title('Sparse solution')
    ax = axes.ravel()[3]
    ax.imshow(convop(solpfw).reshape(imshape), origin='lower', cmap='viridis', norm=normconv)
    ax.set_title('Convolved solution')
    fig.suptitle("Regular PFW")
    plt.show()

    fig = plt.figure(figsize=(11, 11))
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    ax = axes.ravel()[0]
    ax.imshow(x, origin='lower', cmap='Greys', norm=normsparse)
    ax.set_title('Sparse source')
    ax = axes.ravel()[1]
    ax.imshow(g.reshape(imshape), origin='lower', cmap='viridis', norm=normconv)
    ax.scatter(np.unravel_index(samples, imshape)[1], np.unravel_index(samples, imshape)[0], marker='x', color='r')
    ax.set_title('Convolved source')
    ax = axes.ravel()[2]
    ax.imshow(sollm.reshape(imshape), origin='lower', cmap='Greys', norm=normsparse)
    ax.set_title('Sparse solution')
    ax = axes.ravel()[3]
    ax.imshow(convop(sollm).reshape(imshape), origin='lower', cmap='viridis', norm=normconv)
    ax.set_title('Convolved solution')
    fig.suptitle("LocalMaxPFW")
    plt.show()

    ## Compare the different reconstructions with some metrics:
    ### L1 and L2 errors on the components and the convolved images
    ### Reconstruction duration, value of the objective function, dcv at convergence

    # L1 and L2 errors
    l1 = lambda x, y: np.sum(np.abs(x - y))
    l2 = lambda x, y: np.sum((x - y) ** 2)

    l1_x_solpfw = l1(x.ravel(), solpfw)
    l1_x_sollm = l1(x.ravel(), sollm)
    l1_g_solpfw = l1(g, convop(solpfw))
    l1_g_sollm = l1(g, convop(sollm))

    l2_x_solpfw = l2(x.ravel(), solpfw)
    l2_x_sollm = l2(x.ravel(), sollm)
    l2_g_solpfw = l2(g, convop(solpfw))
    l2_g_sollm = l2(g, convop(sollm))

    print(f"L1 error on the sparse source: PFW {l1_x_solpfw:.2e}, LocalMax {l1_x_sollm:.2e}")
    print(f"L1 error on the convolved source: PFW {l1_g_solpfw:.2e}, LocalMax {l1_g_sollm:.2e}")
    print(f"L2 error on the sparse source: PFW {l2_x_solpfw:.2e}, LocalMax {l2_x_sollm:.2e}")
    print(f"L2 error on the convolved source: PFW {l2_g_solpfw:.2e}, LocalMax {l2_g_sollm:.2e}")

    # Duration
    print(f"Duration of PFW: {pfw.stats()[1]['duration'][-1]:.2f}")
    print(f"Duration of LocalMaxPFW: {lmpfw.stats()[1]['duration'][-1]:.2f}")

    # Objective function
    print(f"Objective function at convergence for PFW: {pfw.stats()[1]['Memorize[objective_func]'][-1]:.3e}")
    print(f"Objective function at convergence for LocalMaxPFW: {lmpfw.stats()[1]['Memorize[objective_func]'][-1]:.3e}")

    # DCV
    print(f"DCV at convergence for PFW: {pfw.stats()[1]['AbsError[dcv]'][-1]:.4f}")
    print(f"DCV at convergence for LocalMaxPFW: {lmpfw.stats()[1]['AbsError[dcv]'][-1]:.4f}")

    # Sparsity
    print(f"Sparsity of the solution for PFW: {np.sum(solpfw != 0)}")
    print(f"Sparsity of the solution for LocalMaxPFW: {np.sum(sollm != 0)}")

    certifpfw = fOp.adjoint(y - fOp.apply(solpfw)).reshape(imshape)/lambda_
    certiflm = fOp.adjoint(y - fOp.apply(sollm)).reshape(imshape)/lambda_
    vmax = max(certiflm.max(), certifpfw.max())
    vmin = min(certiflm.min(), certifpfw.min())
    plt.figure(figsize=(11, 5))
    plt.subplot(121)
    plt.imshow(certifpfw, origin='lower', cmap='viridis', vmax=vmax, vmin=vmin)
    plt.contour(certifpfw, levels=[1], colors='r')
    plt.title("Dual certificate PFW")
    plt.subplot(122)
    plt.imshow(certiflm, origin='lower', cmap='viridis', vmax=vmax, vmin=vmin)
    plt.contour(certiflm, levels=[1], colors='r')
    plt.title("Dual certificate LM")
    plt.show()

    ## Save the iterative empirical dual certificates
    data_dir = os.path.join('/home/jarret/PycharmProjects/pyfw-lasso/dev', 'certificates', '2d')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Regular PFW
    save_dir = os.path.join(data_dir, 'pfw')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.ioff()
    for i in range(pfw._astate['idx']):
        plt.figure()
        plt.imshow(pfw.certif_log['certificate'][i].reshape(imshape), origin='lower', cmap='viridis', interpolation='none')
        plt.colorbar()
        candidates = np.unravel_index(pfw.certif_log['candidates'][i], imshape)
        plt.scatter(candidates[1], candidates[0], marker='.', color='r', s=1)
        plt.title(f"Iteration {i}")
        plt.savefig(os.path.join(save_dir, f"iter_{i}.png"))
        plt.close()

    # LocalMaxPFW
    save_dir = os.path.join(data_dir, 'lm')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(lmpfw._astate['idx']):
        plt.figure()
        plt.imshow(lmpfw.certif_log['certificate'][i].reshape(imshape), origin='lower', cmap='viridis', interpolation='none')
        plt.colorbar()
        candidates = np.unravel_index(lmpfw.certif_log['candidates'][i], imshape)
        plt.scatter(candidates[1], candidates[0], marker='.', color='r', s=1)
        plt.title(f"Iteration {i}")
        plt.savefig(os.path.join(save_dir, f"iter_{i}.png"))
        plt.close()
