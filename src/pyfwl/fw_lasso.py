import datetime as dt
import time
import typing as typ

import numpy as np

import pyfwl.utils as ut
import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.operator as pycop
import pycsou.opt.stop as pycos
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
from pycsou.opt.solver.pgd import PGD

__all__ = [
    "VanillaFWforLasso",
    "PolyatomicFWforLasso",
    "dcvStoppingCrit",
]

class GenericFWforLasso(pycs.Solver):
    r"""
    Base class for Frank-Wolfe algorithms (FW) for the LASSO problem.

    This is an abstract class that cannot be instantiated.
    """

    def __init__(
        self,
        data: pyct.NDArray,
        forwardOp: pyco.LinOp,
        lambda_: float,
        *,
        folder=None,  # typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        stop_rate: int = 1,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        show_progress: bool = True,
        log_var: pyct.VarName = (
            "x",
            "dcv",
        ),
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            stop_rate=stop_rate,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )
        self.lambda_ = pycrt.coerce(lambda_)
        self.forwardOp = forwardOp
        self.data = pycrt.coerce(data)

        self._data_fidelity = (
            0.5 * pycop.SquaredL2Norm(dim=self.forwardOp.shape[0]).argshift(-self.data) * self.forwardOp
        )
        self._penalty = self.lambda_ * pycop.L1Norm(dim=self.forwardOp.shape[1])
        self._bound = 0.5 * pycop.SquaredL2Norm().apply(data)[0] / self.lambda_

    def m_init(self, **kwargs):
        xp = pycu.get_array_module(self.data)
        mst = self._mstate  # shorthand
        mst["x"] = xp.zeros(self.forwardOp.shape[1], dtype=pycrt.getPrecision().value)
        mst["dcv"] = np.inf  # "dual certificate value"

    def default_stop_crit(self) -> pycs.StoppingCriterion:
        stop_crit = pycos.RelError(
            eps=1e-4,
            var="objective_func",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        p: NDArray
            (..., N) solution.
        """
        data, _ = self.stats()
        return data.get("x")

    def objective_func(self) -> pyct.NDArray:
        return self._data_fidelity(self._mstate["x"]) + self._penalty(self._mstate["x"])

    def fit(self, **kwargs):
        """
        Set the default value of ``track_objective`` as True, as opposed to the behavior defined in the base class
        ``Solver``.
        """
        track_objective = kwargs.pop("track_objective", True)
        positivity_c = kwargs.pop("positivity_constraint", False)
        self._astate["positivity_c"] = positivity_c

        l_constant = kwargs.pop("diff_lipschitz", None)
        if l_constant is None:
            lipschitz_time = time.time()
            self._data_fidelity.diff_lipschitz(tol=1e-3)
            print("Computation of diff_lipschitz takes {:.4f}".format(time.time() - lipschitz_time))
        else:
            self._data_fidelity._diff_lipschitz = l_constant
        super().fit(track_objective=track_objective, **kwargs)


class VanillaFWforLasso(GenericFWforLasso):
    r"""
    Vanilla version of the Frank-Wolfe algorithm (FW) specifically design to solve the LASSO problem of the form

    .. math ::

        {\argmin_{\mathbf{x} \in \mathbb{R}^N} \; \frac{1}{2} \lVert \mathbf{y} - \mathbf{G}(\mathbf{x}) \rVert_2^2
        + \lambda \lVert \mathbf{x} \rVert_1 }

    where:

    * :math:`\mathbf{G}:\mathbb{R}^N\rightarrow\mathbb{C}^L` is a *linear* operator, referred to as *forward*
     or *measurement* operator.
    * :math:`\mathbf{y}\in\mathbb{C}^L` is the vector of measurements data.
    * :math:`\lambda>0` is the *regularization* or *penalty* parameter.

    The FW algorithms ensure convergence of the iterates in terms of the value of the objective function with a
    convergence rate of :math:`\mathcal{O}(1/k)`[RevFW]_. Consequently, the default stopping criterion is set as a
    threshold over the relative improvement of the value of the objective function. The algorithm stops if the relative
    improvement is below 1e-4. If this default stopping criterion is used, you must not fit the algorithm with argument
    ``track_objective=False``.

    **Remark:** The array module used in the algorithm iterations is inferred from the module of the input measurements.

    ``VanillaFWforLasso.fit()`` **Parametrisation**

    track_objective: Bool
        Indicator to keep track of the value of the objective function along the iterations.
        This value is optional for Polyatomic FW, but can be used to determine a stopping criterion.
        Default is True, can be set to False to accelerate the solving time.
    """

    def __init__(
        self,
        data: pyct.NDArray,
        forwardOp: pyco.LinOp,
        lambda_: float,
        step_size: str = "optimal",
        *,
        folder=None,  # : typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        stop_rate: int = 1,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 50,
        show_progress: bool = True,
        log_var: pyct.VarName = (
            "x",
            "dcv",
        ),
    ):
        r"""

        Parameters
        ----------
        data: NDArray
            (..., L) measurements term(s) in the LASSO objective function. The type of NDArray defines the module used
            for the iterates.
        forwardOp: LinOp
            (N, L) measurement operator in the LASSO objective function.
        lambda_: float
            Regularisation parameter from the LASSO problem.
        step_size: str
            ["regular", "optimal"] If ``step_size="regular"``, the convex combination weight in the weighting step of
            FW is set according to the default strategy, :math:`\gamma_k = 2/k+2`.
            If ``step_size="regular"``, the convex combination weight :math:`\gamma_k` is computed at each step
            as the optimal step size, for which there exists a close form formulation in the specific case of minimising
            the LASSO objective function.
        """
        super().__init__(
            data=data,
            forwardOp=forwardOp,
            lambda_=lambda_,
            folder=folder,
            exist_ok=exist_ok,
            stop_rate=stop_rate,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )

        try:
            assert step_size in ["optimal", "regular"]
            self.step_size_strategy = step_size
        except:
            raise ValueError(f"step_size must be in ['regular', 'optimal'], got {step_size}.")

    def m_init(self, **kwargs):
        super(VanillaFWforLasso, self).m_init(**kwargs)
        self._mstate["lift_variable"] = 0.0  # kept as a float and not an array.

    def m_step(self):
        mst = self._mstate  # shorthand
        xp = pycu.get_array_module(mst["x"])
        mgrad = -self._data_fidelity.grad(mst["x"])
        if self._astate["positivity_c"]:
            new_ind = xp.argmax(mgrad, axis=-1)
        else:
            new_ind = xp.argmax(xp.abs(mgrad), axis=-1)
        dcv = mgrad[new_ind] / self.lambda_  # float
        mst["dcv"] = dcv

        if self.step_size_strategy == "regular":
            gamma = 2 / (2 + self._astate["idx"])
        elif self.step_size_strategy == "optimal":
            # The optimal value of gamma has a closed form expression, which depends on the dual certificate value.
            gamma = -xp.dot(mgrad, mst["x"]).real
            if abs(dcv) > 1.0:
                gamma += self.lambda_ * (mst["lift_variable"] + (abs(dcv) - 1.0) * self._bound)
                injection = pycop.SubSample(self.forwardOp.shape[1], xp.array(new_ind)).T
                gamma /= pycop.SquaredL2Norm().apply(
                    self._bound * np.sign(dcv) * self.forwardOp(injection(xp.array(1.0))) - self.forwardOp(mst["x"])
                )[
                    0
                ]  # we can use numpy (np) as dcv is a float
            else:
                gamma += self.lambda_ * mst["lift_variable"]
                gamma /= pycop.SquaredL2Norm().apply(self.forwardOp(mst["x"]))[0]

        if not 0 < gamma < 1:
            print("Warning, gamma value not valid: {}".format(gamma))
            gamma = np.clip(gamma, 0.0, 1.0)

        mst["x"] *= 1 - gamma
        mst["lift_variable"] *= 1 - gamma
        if abs(dcv) > 1.0:
            mst["x"][new_ind] += pycrt.coerce(gamma * np.sign(dcv) * self._bound)
            mst["lift_variable"] += gamma * self._bound


class PolyatomicFWforLasso(GenericFWforLasso):
    r"""
    Polyatomic version of the Frank-Wolfe algorithm (FW) specifically design to solve the LASSO problem of the form

    .. math ::

        {\argmin_{\mathbf{x} \in \mathbb{R}^N} \; \frac{1}{2} \lVert \mathbf{y} - \mathbf{G}(\mathbf{x}) \rVert_2^2
        + \lambda \lVert \mathbf{x} \rVert_1 }

    where:

    * :math:`\mathbf{G}:\mathbb{R}^N\rightarrow\mathbb{C}^L` is a *linear* operator, referred to as *forward*
     or *measurement* operator.
    * :math:`\mathbf{y}\in\mathbb{C}^L` is the vector of measurements data.
    * :math:`\lambda>0` is the *regularization* or *penalty* parameter.

    This algorithm is presented in the paper [PFW]_ as an improvement over the Vanilla FW algorithm in order to
    converge faster to a solution. Compared to other LASS solvers, it is particularly efficient when the solution is
    supposed to be very sparse, in terms of convergence time as well as memory requirements.

    The FW algorithms ensure convergence of the iterates in terms of the value of the objective function with a
    convergence rate of :math:`\mathcal{O}(1/k)` [RevFW]_. Consequently, the default stopping criterion is set as a
    threshold over the relative improvement of the value of the objective function. The algorithm stops if the relative
    improvement is below 1e-4. If this default stopping criterion is used, you must not fit the algorithm with argument
    ``track_objective=False``.

    **Remark 1:** The array module used in the algorithm iterations is inferred from the module of the input
        measurements.

    **Remark 2:** The second step of the PFW algorithm consists in computing/updating the weights attributed to the
    atoms. To do so, the algorithm solves a LASSO problem with a restricted support defined by the selected atoms.
    The implementation provided for this step makes use of an injection/sub-sampling operator that allows to extract
    the sub-problem of interest. This generic procedure might be sub-optimal and could be improved depending on the
    forward operator (e.g. non-uniform FFT as a restricted support FFT). The interested user can provide an
    alternative and posisbly more efficient method for the correction of the weights by overriding the
    ``.rs_correction()`` method in a child class.

    ``PolyatomicFWforLasso.fit()`` **Parametrisation**

    track_objective: Bool
        Indicator to keep track of the value of the objective function along the iterations.
        This value is optional for Polyatomic FW, but can be used to determine a stopping criterion.
        Default is True, can be set to False to accelerate the solving time.
    """

    def __init__(
        self,
        data: pyct.NDArray,
        forwardOp: pyco.LinOp,
        lambda_: float,
        ms_threshold: float = 0.7,  # multi spikes threshold at init
        init_correction_prec: float = 0.2,
        final_correction_prec: float = 1e-4,
        remove_positions: bool = False,
        min_correction_steps: int = 5,
        *,
        folder=None,  # : typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        stop_rate: int = 1,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 10,
        show_progress: bool = True,
        log_var: pyct.VarName = (
            "x",
            "dcv",
        ),
    ):

        r"""

        Parameters
        ----------
        data: NDArray
            (..., L) measurements term(s) in the LASSO objective function.
        forwardOp: LinOp
            (N, L) measurement operator in the LASSO objective function.
        lambda_: float
            Regularisation parameter from the LASSO problem.
        ms_threshold: float
            Initial threshold for identifying the first atoms, given as a rate of the dual certificate value. This
            parameter impacts the number of atomes chosen at the first step, but also at all the following iterations.
            Low `ms_threshold` value implies a lot of flexiility in the choice of the atoms, whereas high value leads
            to restrictive selection and thus very few atomes at each iteration.
        init_correction_prec: float
            Precision of the first stopping criterion for the step of correction of the weights at first iteration.
        final_correction_prec: float
            Lower bound for the precision of the correction steps. When reached, the precision does no longer decrease
            with the iterations.
        remove_positions: bool
            When set to `True`, the atoms that get attributed a null weight after the correction of the weights are
            removed from the current iterate. When the parameter is set to `False` however, these atoms remain in the
            set of active indices and thus can still be attributed weight at later iterations.
        """
        self._ms_threshold = ms_threshold
        self._init_correction_prec = init_correction_prec
        self._final_correction_prec = final_correction_prec
        self._remove_positions = remove_positions
        self._min_correction_steps = min_correction_steps
        super().__init__(
            data=data,
            forwardOp=forwardOp,
            lambda_=lambda_,
            folder=folder,
            exist_ok=exist_ok,
            stop_rate=stop_rate,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )

    def fit(self, **kwargs):
        self._astate["lock"] = kwargs.pop("lock_reweighting", False)
        super().fit(**kwargs)


    def m_init(self, **kwargs):
        super(PolyatomicFWforLasso, self).m_init(**kwargs)
        xp = pycu.get_array_module(self._mstate["x"])
        mst = self._mstate
        mst["positions"] = xp.array([], dtype="int32")
        # QfR: mst["positions"] stores the set of actives indices of the solution. As the solution should be sparse,
        # in most of the cases this set will remain of relatively small size. Is there any advantage/disadvantage
        # storing it as a np array instead of a xp array ?
        mst["delta"] = None  # initial buffer for multi spike thresholding
        mst["correction_prec"] = self._init_correction_prec

    def m_step(self):
        mst = self._mstate  # shorthand
        mgrad = -self._data_fidelity.grad(mst["x"]) / self.lambda_
        if self._astate["positivity_c"]:
            mst["dcv"] = mgrad.max()
            maxi = mst["dcv"]
        else:
            mst["dcv"] = max(mgrad.max(), mgrad.min(), key=abs)  # float
            # mst["dcv"] is stored as a float in this case and does not handle stacked multidimensional inputs.
            maxi = abs(mst["dcv"])
        if self._astate["idx"] == 1:
            mst["delta"] = maxi * (1.0 - self._ms_threshold)
            mst["N_indices"] = []
            mst["N_candidates"] = []
            mst["correction_iterations"] = []
            mst["correction_durations"] = []
        thresh = maxi - (2 / (self._astate["idx"] + 1)) * mst["delta"]
        if self._astate["positivity_c"]:
            new_indices = (mgrad > max(thresh, 1.0)).nonzero()[0]
        else:
            new_indices = (abs(mgrad) > max(thresh, 1.0)).nonzero()[0]
        mst["N_candidates"].append(new_indices.size)

        xp = pycu.get_array_module(mst["x"])
        if new_indices.size > 0:
            if self._astate["idx"] > 1 and self._remove_positions:
                mst["positions"] = xp.unique(xp.hstack([mst["x"].nonzero()[0], new_indices]))
            else:
                mst["positions"] = xp.unique(xp.hstack([mst["positions"], new_indices]))
        elif self._remove_positions:
            mst["positions"] = (mst["x"] > 1e-5).nonzero()[0]
        # else would correspond to empty new_indices, in this case the set of active indices does not change
        mst["N_indices"].append(mst["positions"].size)

        # print(mst["positions"][:30])

        mst["correction_prec"] = max(self._init_correction_prec / self._astate["idx"], self._final_correction_prec)
        if mst["positions"].size > 1:
            mst["x"] = self.rs_correction(mst["positions"], self._mstate["correction_prec"])
        elif mst["positions"].size == 1:  # case 1-sparse solution => the solution can be computed explicitly
            tmp = xp.zeros(self.forwardOp.shape[1], dtype=pycrt.getPrecision().value)
            tmp[mst["positions"]] = 1.0
            column = self.forwardOp(tmp)
            corr = xp.dot(self.data, column).real
            if abs(corr) <= self.lambda_:
                mst["x"] = xp.zeros(self.forwardOp.shape[1], dtype=pycrt.getPrecision().value)
            elif corr > self.lambda_:
                mst["x"] = ((corr - self.lambda_) / pycop.SquaredL2Norm().apply(column)[0]) * tmp
            else:
                mst["x"] = ((corr + self.lambda_) / pycop.SquaredL2Norm().apply(column)[0]) * tmp
        else:
            mst["x"] = xp.zeros(self.forwardOp.shape[1], dtype=pycrt.getPrecision().value)

    def rs_correction(self, support_indices: pyct.NDArray, precision: float) -> pyct.NDArray:
        r"""
        Method to update the weights after the selection of the new atoms. It solves a LASSO problem with a
        restricted support corresponding to the current set of active indices (active atoms). As mentioned,
        this method should be overriden in a child class for case-specific improved implementation.

        Parameters
        ----------
        support_indices: NDArray
            Set of active indices.
        precision: float
            Value of the stopping criterion for the correction.

        Returns
        -------
        weights: NDArray
            New iterate with the updated weights.
        """

        def correction_stop_crit(eps) -> pycs.StoppingCriterion:
            stop_crit = pycos.RelError(
                eps=eps,
                var="x",
                f=None,
                norm=2,
                satisfy_all=True,
            )
            return stop_crit

        subsample = pycop.SubSample(self.forwardOp.shape[1], support_indices)
        # print(injection.shape)
        # rs_data_fid = self._data_fidelity * injection
        # rs_data_fid.diff_lipschitz()
        rs_data_fid = self.rs_data_fid(support_indices)
        x0 = subsample(self._mstate["x"])
        if self._astate["positivity_c"]:
            penalty = ut.L1NormPositivityConstraint(shape=(1, None))
        else:
            penalty = pycop.L1Norm()
        apgd = PGD(rs_data_fid, self.lambda_ * penalty, show_progress=False)
        # The penalty is agnostic to the dimension in this implementation (L1Norm()).
        stop = pycos.MaxIter(n=self._min_correction_steps)
        if not self._astate["lock"]:
            stop &= correction_stop_crit(precision)
        apgd.fit(
            x0=x0,
            tau=1 / self._data_fidelity._diff_lipschitz,
            stop_crit=(stop | pycos.MaxDuration(t=dt.timedelta(seconds=1000)))
        )
        self._mstate["correction_iterations"].append(apgd.stats()[1]["iteration"][-1])
        self._mstate["correction_durations"].append(apgd.stats()[1]["duration"][-1])
        sol, _ = apgd.stats()
        return subsample.adjoint(sol["x"])

    def rs_data_fid(self, support_indices: pyct.NDArray) -> pyco.DiffFunc:
        injection = pycop.SubSample(self.forwardOp.shape[1], support_indices).T
        return 0.5 * pycop.SquaredL2Norm(dim=self.forwardOp.shape[0]).argshift(-self.data) * self.forwardOp * injection

    def post_process(self):
        mst = self._mstate
        mst["x"] = self.rs_correction(mst["positions"], self._final_correction_prec)


def dcvStoppingCrit(eps: float = 1e-2) -> pycs.StoppingCriterion:
    r"""
    Instantiate a ``StoppingCriterion`` class based on the maximum value of the dual certificate, that is automatically
    computed along the iterations of any Frank-Wolfe algorithm. At convergence, the maximum absolute value component of
    the dual certificate is 1, and thus tracking the difference to 1 provides us with a natural stopping criterion.

    Parameters
    ----------
    eps: float
        Precision on the distance to 1 for stopping.

    Returns
    -------
    stop_crit: pycs.StoppingCriterion
        An instance of the ``StoppingCriterion`` class that can be fed to the ``.fit()`` method of any
        Frank-Wolfe solver.
    """

    def abs_diff_to_one(x):
        return abs(x) - 1.0

    stop_crit = pycos.AbsError(
        eps=eps,
        var="dcv",
        f=abs_diff_to_one,
        norm=2,
        satisfy_all=True,
    )
    return stop_crit
