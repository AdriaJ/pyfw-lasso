"""
The solutions are exactly similar with or without the sparse implementation of the operator, so the sparse
implementation is correct and the test is successful.
"""
import datetime as dt
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pycsou.abc as pyca
import pycsou.operator as pycop
import pycsou.opt.stop as pycos
import pycsou.util.ptype as pyct
from pycsou.opt.solver.pgd import PGD

import pyfwl

# matplotlib.use('TkAgg')
matplotlib.use("Qt5Agg")

seed = None  # for reproducibility

# Dimensions of the problem
L = 1000
N = 10000
k = 150
psnr = 20

# Parameters for reconstruction
lambda_factor = 0.1

# Parameters of the solver
remove = False
eps = 1e-4
min_iterations = 10
tmax = 15.0
ms_threshold = 0.8
init_correction = 1e-1
final_correction = 1e-6
eps_dcv = 1e-2

stop_crit = pycos.RelError(
    eps=eps,
    var="objective_func",
    f=None,
    norm=2,
    satisfy_all=True,
)
# alternative stopping criteria
dcv = pyfwl.dcvStoppingCrit(eps_dcv)
# Minimum number of iterations
min_iter = pycos.MaxIter(n=min_iterations)

# track DCV
track_dcv = pycos.AbsError(eps=1e-10, var="dcv", f=None, norm=2, satisfy_all=True)

"""
We consider a LASSO problem with an explicit matrix as a forward operator. In this case, we can accelerate the solving
with PFW by providing an efficient implementation of the forward operator when applied to a sparse vector.
We simply do that by slicing the matrix.
"""


class SparseVFW(pyfwl.VFWLasso):
    def rs_forwardOp(self, support_indices: pyct.NDArray) -> pyca.LinOp:
        assert self.forwardOp._name == "_ExplicitLinOp"
        if support_indices.size > 0:
            return pyca.LinOp.from_array(self.forwardOp.mat[:, support_indices])
        else:
            return pycop.NullOp(shape=(1, None))


class SparsePFW(pyfwl.PFWLasso):
    def rs_forwardOp(self, support_indices: pyct.NDArray) -> pyca.LinOp:
        assert self.forwardOp._name == "_ExplicitLinOp"
        if support_indices.size > 0:
            return pyca.LinOp.from_array(self.forwardOp.mat[:, support_indices])
        else:
            return pycop.NullOp(shape=(1, None))


if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed)

    mat = rng.normal(size=(L, N))  # forward matrix
    indices = rng.choice(N, size=k)  # indices of active components in the source
    injection = pycop.SubSample(N, indices).T
    source = injection(rng.normal(size=k))  # sparse source

    op = pyca.LinOp.from_array(mat)
    lip = op.lipschitz()
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
    vfw.fit(stop_crit=(min_iter & stop_crit) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)) | track_dcv,
            diff_lipschitz=lip ** 2)
    data_v, hist_v = vfw.stats()
    time_v = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_v))

    print("Polyatomic FW: Solving ...")
    start = time.time()
    pfw.fit(stop_crit=(min_iter & stop_crit) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)) | track_dcv,
            diff_lipschitz=lip ** 2)
    data_p, hist_p = pfw.stats()
    time_p = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_p))

    # Solving the same problems with more efficient computation of the forward op
    svfw = SparseVFW(
        measurements,
        op,
        lambda_,
        step_size="optimal",
        show_progress=False
    )
    spfw = SparsePFW(
        measurements,
        op,
        lambda_,
        ms_threshold=ms_threshold,
        remove_positions=remove,
        show_progress=False,
        init_correction_prec=init_correction,
        final_correction_prec=final_correction,
    )

    print("Sparse Vanilla FW: Solving ...")
    start = time.time()
    svfw.fit(stop_crit=(min_iter & stop_crit) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)) | track_dcv,
             diff_lipschitz=lip ** 2)
    data_sv, hist_sv = svfw.stats()
    time_sv = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_sv))

    print("Sparse Polyatomic FW: Solving ...")
    start = time.time()
    spfw.fit(stop_crit=(min_iter & stop_crit) | pycos.MaxDuration(t=dt.timedelta(seconds=tmax)) | track_dcv,
             diff_lipschitz=lip ** 2)
    data_sp, hist_sp = spfw.stats()
    time_sp = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_sp))

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
        tau=1 / lip ** 2,
    )
    data_apgd, hist_apgd = pgd.stats()
    time_pgd = time.time() - start
    print("\tSolved in {:.3f} seconds".format(time_pgd))

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(pfw._mstate["N_indices"], label="Support size")
    # plt.plot(pfw._mstate["N_candidates"], label="candidates")
    # plt.legend()
    # plt.subplot(122)
    # plt.plot(pfw._mstate["correction_iterations"], label="iterations")
    # plt.ylim(bottom=0.0)
    # plt.twinx()
    # plt.plot(pfw._mstate["correction_durations"], label="duration", c="r")
    # plt.ylim(bottom=0.0)
    # plt.title("Correction iterations")
    # plt.legend()
    # plt.show()

    apgd_dcv = np.abs(op.adjoint(measurements - op(data_apgd["x"]))).max() / lambda_
    print("Final value of dual certificate:\n\tVFW: {:.4f}\n\tPFW: {:.4f}\n\tsVFW: {:.4f}\n\tsPFW: {:.4f}\n\tAPGD: {"
          ":.4f}".format(data_v["dcv"],
                         data_p["dcv"],
                         data_sv["dcv"],
                         data_sp["dcv"],
                         apgd_dcv))
    print(
        "Final value of objective:\n\tVFW : {:.4f}\n\tPFW : {:.4f}\n\tsVFW: {:.4f}\n\tsPFW: {:.4f}\n\tAPGD: {:.4f}".format(
            hist_v[-1][-1], hist_p[-1][-1], hist_sv[-1][-1], hist_sp[-1][-1], hist_apgd[-1][-1]
        )
    )

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(pfw._mstate["N_indices"], label="Support size")
    # plt.plot(pfw._mstate["N_candidates"], label="candidates")
    # plt.legend()
    # plt.subplot(122)
    # plt.plot(pfw._mstate["correction_iterations"], label="iterations")
    # plt.ylim(bottom=0.0)
    # plt.twinx()
    # plt.plot(pfw._mstate["correction_durations"], label="duration", c="r")
    # plt.ylim(bottom=0.0)
    # plt.title("Correction iterations")
    # plt.legend()
    # plt.show()

    def indices(arr):
        return np.where(np.abs(arr) > 1e-4)[0]


    # plt.figure(figsize=(10, 8))
    # plt.suptitle("Compare the reconstructions")
    # plt.subplot(211)
    # plt.stem(indices(source), source[indices(source)], label="source", linefmt="C0-", markerfmt="C0o")
    # plt.stem(indices(data_apgd["x"]), data_apgd["x"][indices(data_apgd["x"])], label="apgd", linefmt="C1:", markerfmt="C1x")
    # plt.stem(indices(data_v["x"]), data_v["x"][indices(data_v["x"])], label="VFW", linefmt="C2:", markerfmt="C2x")
    # plt.stem(indices(data_p["x"]), data_p["x"][indices(data_p["x"])], label="PFW", linefmt="C3:", markerfmt="C3x")
    # plt.legend()
    # plt.title("Relative improvement as stopping criterion")
    # plt.subplot(212)
    # plt.stem(indices(source), source[indices(source)], label="source", linefmt="C0-", markerfmt="C0o")
    # plt.stem(indices(data_apgd["x"]), data_apgd["x"][indices(data_apgd["x"])], label="apgd", linefmt="C1:", markerfmt="C1x")
    # plt.stem(indices(data_v_dcv["x"]), data_v_dcv["x"][indices(data_v_dcv["x"])], label="VFW", linefmt="C2:", markerfmt="C2x")
    # plt.stem(indices(data_p_dcv["x"]), data_p_dcv["x"][indices(data_p_dcv["x"])], label="PFW", linefmt="C3:", markerfmt="C3x")
    # plt.legend()
    # plt.title("Dual certificate value as stopping criterion")
    # plt.show()

    mini = min(
        hist_p["Memorize[objective_func]"][-1],
        hist_v["Memorize[objective_func]"][-1],
        hist_apgd["Memorize[objective_func]"][-1],
        hist_sp["Memorize[objective_func]"][-1],
        hist_sv["Memorize[objective_func]"][-1],
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
        hist_sp["duration"],
        (hist_sp["Memorize[objective_func]"] - mini) / (hist_apgd["Memorize[objective_func]"][0]),
        label="Sparse-PFW",
        marker="+",
    )
    plt.plot(
        hist_sv["duration"],
        (hist_sv["Memorize[objective_func]"] - mini) / (hist_apgd["Memorize[objective_func]"][0]),
        label="Sparse-VFW",
    )
    plt.plot(
        hist_apgd["duration"],
        (hist_apgd["Memorize[objective_func]"] - mini) / (hist_apgd["Memorize[objective_func]"][0]),
        label="APGD",
        marker="x",
    )
    plt.legend()
    plt.ylabel("OFV")
    plt.title("Log-scale plot")

    plt.subplot(212)
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
        hist_sp["duration"],
        (hist_sp["Memorize[objective_func]"] - mini) / (hist_apgd["Memorize[objective_func]"][0]),
        label="Sparse-PFW",
        marker="+",
    )
    plt.plot(
        hist_sv["duration"],
        (hist_sv["Memorize[objective_func]"] - mini) / (hist_apgd["Memorize[objective_func]"][0]),
        label="Sparse-VFW",
    )
    plt.plot(
        hist_apgd["duration"],
        (hist_apgd["Memorize[objective_func]"] - mini) / (hist_apgd["Memorize[objective_func]"][0]),
        label="APGD",
        marker="x",
    )
    plt.legend()
    plt.ylabel("OFV")
    plt.title("Linear scale plot")
    plt.show()

    # plt.figure(figsize=(10, 8))
    # plt.suptitle("Stopping criterion values")
    # plt.subplot(211)
    # plt.yscale("log")
    # plt.plot(hist_p["duration"], hist_p["RelError[objective_func]"], label="PFW", marker="+")
    # plt.plot(hist_v["duration"], hist_v["RelError[objective_func]"], label="VFW")
    # plt.title("Stop: Relative improvement ofv")
    # plt.legend()
    # plt.subplot(212)
    # plt.yscale("log")
    # plt.plot(hist_p_dcv["duration"], hist_p_dcv["AbsError[dcv]"], label="PFW")
    # plt.plot(hist_v_dcv["duration"], hist_v_dcv["AbsError[dcv]"], label="VFW")
    # plt.title("Stop: Absolute error dcv")
    # plt.legend()
    # plt.show()

    # start = time.time()
    # pfw.post_process()
    # print(f"Post-processing in: {time.time()-start:.4f}s")
    # print(f"Final value of the objective: {pfw.objective_func()[0]:.4f}")
