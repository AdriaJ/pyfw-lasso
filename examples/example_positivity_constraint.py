"""
In order to enforce positivity on the solutions provided by the FW algorithms, one simple needs to provide the keyword
`positivity_c` to the `.fit` ffunction with value `True`.
"""

import datetime as dt
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pycsou.abc as pyca
import pycsou.operator as pycop
import pycsou.opt.stop as pycos
from pycsou.opt.solver.pgd import PGD

import pyfwl

matplotlib.use("Qt5Agg")

seed = None  # for reproducibility

# Dimensions of the problem
L = 30
N = 100
k = 10
psnr = 20

# Parameters for reconstruction
lambda_factor = 0.1
remove = True
eps = 1e-5
min_iterations = 100
tmax = 5.0

stop_crit = pycos.RelError(
    eps=eps,
    var="objective_func",
    f=None,
    norm=2,
    satisfy_all=True,
)

# Minimum number of iterations
min_iter = pycos.MaxIter(n=min_iterations)

full_stop = (min_iter & stop_crit) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax))

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed)

    mat = rng.normal(size=(L, N))  # forward matrix
    indices = rng.choice(N, size=k)  # indices of active components in the source
    injection = pycop.SubSample(N, indices).T
    source = np.abs(injection((rng.normal(size=k))))  # sparse source

    op = pyca.LinOp.from_array(mat)
    start = time.time()
    lip = op.lipschitz()
    print("Computation of the Lipschitz constant in {:.2f}".format(time.time()-start))
    noiseless_measurements = op(source)
    std = np.max(np.abs(noiseless_measurements)) * 10 ** (-psnr / 20)
    noise = rng.normal(0, std, size=L)
    measurements = noiseless_measurements + noise

    lambda_ = lambda_factor * np.linalg.norm(op.T(measurements), np.infty)  # rule of thumb to define lambda

    vfw = pyfwl.VFWLasso(measurements, op, lambda_, step_size="optimal", show_progress=False)
    pfw = pyfwl.PFWLasso(
        measurements, op, lambda_, ms_threshold=0.9, remove_positions=remove, show_progress=False
    )

    print("\nVanilla FW: Solving ...")
    start = time.time()
    # vfw.fit(stop_crit=min_iter & vfw.default_stop_crit())
    vfw.fit(stop_crit=full_stop, diff_lipschitz=lip**2)
    data_v, hist_v = vfw.stats()
    time_v = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_v))

    print("Polyatomic FW: Solving ...")
    start = time.time()
    # pfw.fit(stop_crit=min_iter & pfw.default_stop_crit())
    pfw.fit(stop_crit=full_stop, diff_lipschitz=lip**2)
    data_p, hist_p = pfw.stats()
    time_p = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_p))

    # Explicit definition of the objective function for APGD
    data_fid = 0.5 * pycop.SquaredL2Norm(dim=op.shape[0]).argshift(-measurements) * op
    regul = lambda_ * pycop.L1Norm()

    print("Solving with APGD: ...")
    pgd = PGD(data_fid, regul, show_progress=False)
    start = time.time()
    pgd.fit(
        x0=np.zeros(N, dtype="float64"),
        stop_crit=(min_iter & pgd.default_stop_crit()) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)),
        track_objective=True,
        tau=1/lip**2
    )
    data_apgd, hist_apgd = pgd.stats()
    time_pgd = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_pgd))

    apgd_dcv = np.abs(op.adjoint(measurements - op(data_apgd["x"]))).max()/lambda_
    print("Final value of dual certificate:\n\tVFW: {:.4f}\n\tPFW: {:.4f}\n\tAPGD: {:.4f}".format(data_v["dcv"],
                                                                                                  data_p["dcv"],
                                                                                                  apgd_dcv))
    print(
        "Final value of objective:\n\tVFW : {:.4f}\n\tPFW : {:.4f}\n\tAPGD: {:.4f}".format(
            hist_v[-1][-1], hist_p[-1][-1], hist_apgd[-1][-1]
        )
    )

    print("\n\nPositivity constraint:\n")
    print("Vanilla FW: Solving ...")
    # Solving the same problems with positivity constraint
    vfw.fit(stop_crit=full_stop, positivity_constraint=True, diff_lipschitz=lip**2)
    data_v_pos, hist_v_pos = vfw.stats()
    print("\tSolved in {:.3f} seconds".format(hist_v_pos["duration"][-1]))

    print("Polyatomic FW: Solving ...")
    pfw.fit(stop_crit=full_stop, positivity_constraint=True, diff_lipschitz=lip**2)
    data_p_pos, hist_p_pos = pfw.stats()
    print("\tSolved in {:.3f} seconds".format(hist_p_pos["duration"][-1]))

    print("Solving with PGD: ...")
    posRegul = lambda_ * pyfwl.L1NormPositivityConstraint(shape=(1, None))
    pgd_pos = PGD(data_fid, posRegul, show_progress=False)
    pgd_pos.fit(
        x0=np.zeros(N, dtype="float64"),
        stop_crit=(min_iter & pgd_pos.default_stop_crit()) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)),
        track_objective=True,
        tau=1/lip**2,
    )
    # cv = CondatVu(f=data_fid, g=regul, h=pycdevu.NonNegativeOrthant(shape=(1, N)), show_progress=True, verbosity=100)
    # cv.fit(
    #     x0=np.zeros(N, dtype="float64"),
    #     stop_crit=(min_iter & cv.default_stop_crit()) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)),
    #     track_objective=True,
    # )
    # data_pds, hist_pds = cv.stats()
    data_pgd_pos, hist_pgd_pos = pgd_pos.stats()
    print("\tSolved in {:.3f} seconds".format(hist_pgd_pos["duration"][-1]))

    apgd_pos_dcv = np.abs(op.adjoint(measurements - op(data_apgd["x"]))).max()/lambda_
    print("Final value of dual certificate:\n\tVFW: {:.4f}\n\tPFW: {:.4f}\n\tAPGD: {:.4f}".format(data_v_pos["dcv"],
                                                                                                  data_p_pos["dcv"],
                                                                                                  apgd_pos_dcv))
    print(
        "Final value of objective:\n\tVFW : {:.4f}\n\tPFW : {:.4f}\n\tAPGD: {:.4f}".format(
            hist_v_pos[-1][-1], hist_p_pos[-1][-1], hist_pgd_pos[-1][-1]
        )
    )
    ###########################################################################################
    def supp(arr):
        return np.where(np.abs(arr) > 1e-4)[0]

    plt.figure(figsize=(10, 8))
    plt.suptitle("Compare the reconstructions")
    plt.subplot(211)
    plt.stem(supp(source), source[supp(source)], label="source", linefmt="C0-", markerfmt="C0o")
    plt.stem(supp(data_apgd["x"]), data_apgd["x"][supp(data_apgd["x"])], label="APGD", linefmt="C1:", markerfmt="C1x")
    plt.stem(supp(data_v["x"]), data_v["x"][supp(data_v["x"])], label="VFW", linefmt="C2:", markerfmt="C2x")
    plt.stem(supp(data_p["x"]), data_p["x"][supp(data_p["x"])], label="PFW", linefmt="C3:", markerfmt="C3x")
    plt.legend()
    plt.title("Unconstrained reconstruction")
    plt.subplot(212)
    plt.stem(supp(source), source[supp(source)], label="source", linefmt="C0-", markerfmt="C0o")
    plt.stem(supp(data_pgd_pos["x"]), data_pgd_pos["x"][supp(data_pgd_pos["x"])], label="APGD", linefmt="C1:", markerfmt="C1x")
    plt.stem(supp(data_v_pos["x"]), data_v_pos["x"][supp(data_v_pos["x"])], label="VFW", linefmt="C2:", markerfmt="C2x")
    plt.stem(supp(data_p_pos["x"]), data_p_pos["x"][supp(data_p_pos["x"])], label="PFW", linefmt="C3:", markerfmt="C3x")
    plt.legend()
    plt.title("Positivity constrained solution")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.suptitle("Objective function values")
    plt.subplot(211)
    plt.yscale("log")
    plt.plot(hist_p["duration"], hist_p["Memorize[objective_func]"], label="PFW")
    plt.plot(hist_v["duration"], hist_v["Memorize[objective_func]"], label="VFW")
    plt.plot(hist_apgd["duration"], hist_apgd["Memorize[objective_func]"], label="APGD")
    plt.legend()
    plt.ylabel("OFV")
    plt.title("Unconstrained reconstruction")
    plt.subplot(212)
    plt.yscale("log")
    plt.plot(hist_p_pos["duration"], hist_p_pos["Memorize[objective_func]"], label="PFW")
    plt.plot(hist_v_pos["duration"], hist_v_pos["Memorize[objective_func]"], label="VFW")
    plt.plot(hist_pgd_pos["duration"], hist_pgd_pos["Memorize[objective_func]"], label="PGD")
    plt.ylabel("OFV")
    plt.legend()
    plt.title("Positivity constrained solution")
    plt.show()
