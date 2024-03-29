import datetime as dt
import time

import matplotlib.pyplot as plt
import numpy as np

import pyxu.abc as pxabc
import pyxu.operator as pxop
import pyfwl
import pyxu.opt.stop as pxos
from pyxu.opt.solver.pgd import PGD

# matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')

seed = None  # for reproducibility

# Dimensions of the problem
L = 1000
N = 10000
k = 150
psnr = 20

# Parameters for reconstruction
lambda_factor = 0.1

# Parameters of the solver
remove = True
eps = 1e-4
min_iterations = 10
tmax = 15.0
ms_threshold = 0.8
init_correction = 1e-1
final_correction = 1e-6
eps_dcv = 1e-2


stop_crit = pxos.RelError(
    eps=eps,
    var="objective_func",
    f=None,
    norm=2,
    satisfy_all=True,
)
# alternative stopping criteria
dcv = pyfwl.dcvStoppingCrit(eps_dcv)
# Minimum number of iterations
min_iter = pxos.MaxIter(n=min_iterations)

# track DCV
track_dcv = pxos.AbsError(eps=1e-10, var="dcv", f=None, norm=2, satisfy_all=True)

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed)

    mat = rng.normal(size=(L, N))  # forward matrix
    indices = rng.choice(N, size=k)  # indices of active components in the source
    injection = pxop.SubSample(N, indices).T
    source = injection(rng.normal(size=k))  # sparse source

    op = pxabc.LinOp.from_array(mat)
    op.lipschitz = op.estimate_lipschitz(method='svd', tol=1.e-2)
    noiseless_measurements = op(source)
    std = np.max(np.abs(noiseless_measurements)) * 10 ** (-psnr / 20)
    noise = rng.normal(0, std, size=L)
    measurements = noiseless_measurements + noise

    lambda_ = lambda_factor * np.linalg.norm(op.T(measurements), np.infty)  # rule of thumb to define lambda

    vfw = pyfwl.VFWLasso(
        measurements,
        op,
        lambda_,
        step_size="optimal",
        show_progress=False
    )
    pfw = pyfwl.PFWLasso(
        measurements,
        op,
        lambda_,
        ms_threshold=ms_threshold,
        remove_positions=remove,
        show_progress=False,
        init_correction_prec=init_correction,
        final_correction_prec=final_correction,
    )

    print("\nVanilla FW: Solving ...")
    start = time.time()
    vfw.fit(stop_crit=(min_iter & stop_crit) | pxos.MaxDuration(t=dt.timedelta(seconds=tmax)) | track_dcv,
            diff_lipschitz=op.lipschitz**2)
    data_v, hist_v = vfw.stats()
    time_v = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_v))

    print("Polyatomic FW: Solving ...")
    start = time.time()
    pfw.fit(stop_crit=(min_iter & stop_crit) | pxos.MaxDuration(t=dt.timedelta(seconds=tmax)) | track_dcv,
            diff_lipschitz=op.lipschitz**2)
    data_p, hist_p = pfw.stats()
    time_p = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_p))

    # Explicit definition of the objective function for APGD
    data_fid = 0.5 * pxop.SquaredL2Norm(dim=op.shape[0]).argshift(-measurements) * op
    regul = lambda_ * pxop.L1Norm(N)

    print("Solving with APGD: ...")
    pgd = PGD(data_fid, regul, show_progress=False)
    start = time.time()
    pgd.fit(
        x0=np.zeros(N, dtype="float64"),
        stop_crit=(min_iter & pgd.default_stop_crit()) | pxos.MaxDuration(t=dt.timedelta(seconds=tmax)),
        track_objective=True,
        tau=1/op.lipschitz**2,
    )
    data_apgd, hist_apgd = pgd.stats()
    time_pgd = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_pgd))

    pfw.diagnostics()

    apgd_dcv = np.abs(op.adjoint(measurements - op(data_apgd["x"]))).max()/lambda_
    print("Final value of dual certificate:\n\tVFW: {:.4f}\n\tPFW: {:.4f}\n\tAPGD: {:.4f}".format(data_v["dcv"],
                                                                                                  data_p["dcv"],
                                                                                                  apgd_dcv))
    print(
        "Final value of objective:\n\tAPGD: {:.4f}\n\tVFW : {:.4f}\n\tPFW : {:.4f}".format(
            hist_apgd[-1][-1], hist_v[-1][-1], hist_p[-1][-1]
        )
    )

    # Solving the same problems with another stopping criterion: DCV
    vfw.fit(stop_crit=(min_iter & dcv) | pxos.MaxDuration(t=dt.timedelta(seconds=tmax)),
            diff_lipschitz=op.lipschitz**2)
    data_v_dcv, hist_v_dcv = vfw.stats()

    pfw.fit(stop_crit=(min_iter & dcv) | pxos.MaxDuration(t=dt.timedelta(seconds=tmax)),
            diff_lipschitz=op.lipschitz**2)
    data_p_dcv, hist_p_dcv = pfw.stats()

    def indices(arr):
        return np.where(np.abs(arr) > 1e-4)[0]

    plt.figure(figsize=(10, 8))
    plt.suptitle("Compare the reconstructions")
    plt.subplot(211)
    plt.stem(indices(source), source[indices(source)], label="source", linefmt="C0-", markerfmt="C0o")
    plt.stem(indices(data_apgd["x"]), data_apgd["x"][indices(data_apgd["x"])], label="apgd", linefmt="C1:", markerfmt="C1x")
    plt.stem(indices(data_v["x"]), data_v["x"][indices(data_v["x"])], label="VFW", linefmt="C2:", markerfmt="C2x")
    plt.stem(indices(data_p["x"]), data_p["x"][indices(data_p["x"])], label="PFW", linefmt="C3:", markerfmt="C3x")
    plt.legend()
    plt.title("Relative improvement as stopping criterion")
    plt.subplot(212)
    plt.stem(indices(source), source[indices(source)], label="source", linefmt="C0-", markerfmt="C0o")
    plt.stem(indices(data_apgd["x"]), data_apgd["x"][indices(data_apgd["x"])], label="apgd", linefmt="C1:", markerfmt="C1x")
    plt.stem(indices(data_v_dcv["x"]), data_v_dcv["x"][indices(data_v_dcv["x"])], label="VFW", linefmt="C2:", markerfmt="C2x")
    plt.stem(indices(data_p_dcv["x"]), data_p_dcv["x"][indices(data_p_dcv["x"])], label="PFW", linefmt="C3:", markerfmt="C3x")
    plt.legend()
    plt.title("Dual certificate value as stopping criterion")
    plt.show()

    mini = min(
        hist_p["Memorize[objective_func]"][-1],
        hist_v["Memorize[objective_func]"][-1],
        hist_apgd["Memorize[objective_func]"][-1],
    )
    plt.figure(figsize=(10, 8))
    plt.suptitle("Objective function values")
    plt.subplot(211)
    plt.yscale("log")
    plt.plot(
        hist_p["duration"],
        (hist_p["Memorize[objective_func]"] - mini) / (hist_apgd["Memorize[objective_func]"][0]),
        label="PFW",
        marker="+",
    )
    plt.plot(
        hist_v["duration"],
        (hist_v["Memorize[objective_func]"] - mini) / (hist_apgd["Memorize[objective_func]"][0]),
        label="VFW",
    )
    plt.plot(
        hist_apgd["duration"],
        (hist_apgd["Memorize[objective_func]"] - mini) / (hist_apgd["Memorize[objective_func]"][0]),
        label="APGD",
        marker="x",
    )
    plt.legend()
    plt.ylabel("OFV")
    plt.title("Stop: Relative improvement ofv")

    plt.subplot(212)
    plt.yscale("log")
    plt.plot(hist_p_dcv["duration"], hist_p_dcv["Memorize[objective_func]"], label="PFW")
    plt.plot(hist_v_dcv["duration"], hist_v_dcv["Memorize[objective_func]"], label="VFW")
    plt.plot(hist_apgd["duration"], hist_apgd["Memorize[objective_func]"], label="APGD")
    plt.ylabel("OFV")
    plt.legend()
    plt.title("Stop: Absolute error dcv")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.suptitle("Stopping criterion values")  # TO CHAAAAAAAAAAANGE + time x-axis + push
    plt.subplot(211)
    plt.yscale("log")
    plt.plot(hist_p["duration"], hist_p["RelError[objective_func]"], label="PFW", marker="+")
    plt.plot(hist_v["duration"], hist_v["RelError[objective_func]"], label="VFW")
    plt.title("Stop: Relative improvement ofv")
    plt.legend()
    plt.subplot(212)
    plt.yscale("log")
    plt.plot(hist_p_dcv["duration"], hist_p_dcv["AbsError[dcv]"], label="PFW")
    plt.plot(hist_v_dcv["duration"], hist_v_dcv["AbsError[dcv]"], label="VFW")
    plt.title("Stop: Absolute error dcv")
    plt.legend()
    plt.show()

    start = time.time()
    pfw.post_process()
    print(f"Post-processing in: {time.time()-start:.4f}s")
    print(f"Final value of the objective: {pfw.objective_func()[0]:.4f}")