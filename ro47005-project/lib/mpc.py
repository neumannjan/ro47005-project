"""
Path tracking simulation with iterative linear model predictive control for speed and steer control
author: Atsushi Sakai (@Atsushi_twi)
"""
import math
import sys
from typing import Tuple, List, Optional

import cvxpy
import numpy as np

from lib.simulation import State, update_state, DT, WB, MAX_SPEED, MIN_SPEED, MAX_STEER

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 14  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q_v_yaw = np.diag([0., 0.5])  # state cost matrix [v, yaw]
Qf = np.diag([1.0, 1.0, 0., 0.5]) * 10.  # state final matrix [x, y, v, yaw]
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 13.0  # max simulation time

# iterative paramter
MAX_ITER = 1  # Max iteration
DU_TH = 0.1  # iteration finish param

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]

MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_ACCEL = 2.0  # maximum accel [m/ss]
MAX_DECEL = -3.5  # maximum deceleration [m/ss]


class MPCSolutionNotFoundException(Exception):
    pass


def _get_linear_model_matrix(v, phi, delta):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def _get_nparray_from_matrix(x):
    return np.array(x).flatten()


def _calc_nearest_index(state, cx, cy, pind):
    dx = cx[pind:] - state.x
    dy = cy[pind:] - state.y

    d = dx ** 2 + dy ** 2
    return np.argmin(d) + pind


def _calc_ref_trajectory(state, cx, cy, cyaw, dl, start_idx, ov):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    start_idx = max(_calc_nearest_index(state, cx, cy, start_idx), start_idx)

    dref[:, :T + 1] = 0.0  # steer operational point should be 0.0 for indices 0 to T

    if ov is None:
        ov = np.ones((T + 1,)) * max(state.v, 5 / 3.6)

    travel = np.cumsum(np.abs(ov) * DT)
    idx = np.rint(travel / dl).astype(int)
    idx = np.minimum(idx + start_idx, ncourse - 1)

    xref[0, :T + 1] = cx[idx]
    xref[1, :T + 1] = cy[idx]
    # we skip the velocity - we don't need it (Q must have a 0 for it)
    xref[3, :T + 1] = cyaw[idx]

    reaches_end = idx == ncourse - 1

    return xref, start_idx, dref, reaches_end


def _predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def _linear_mpc_control(xref, xbar, x0, dref, reaches_end):
    """
    linear mpc control
    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    :param reaches_end:
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T + 1):
        if t > 0:
            if not reaches_end[t]:
                # penalize difference from reference perpendicular to yaw strongly
                ref_yaw_perp = xref[3, t] + 0.5 * np.pi
                cost += cvxpy.quad_form(xref[:2, t] - x[:2, t], _get_xy_cost_mtx_for_orientation(ref_yaw_perp) * 10.)

                # penalize difference from reference parallel to yaw weakly
                ref_yaw = xref[3, t]
                cost += cvxpy.quad_form(xref[:2, t] - x[:2, t], _get_xy_cost_mtx_for_orientation(ref_yaw) * 1.0)

                # penalize velocity and yaw itself
                cost += cvxpy.quad_form(xref[2:, t] - x[2:, t], Q_v_yaw)
            else:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], Qf)

        if t < T:
            cost += cvxpy.quad_form(u[:, t], R)

            A, B, C = _get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [u[0, :] <= MAX_ACCEL]
    constraints += [u[0, :] >= MAX_DECEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = _get_nparray_from_matrix(x.value[0, :])
        oy = _get_nparray_from_matrix(x.value[1, :])
        ov = _get_nparray_from_matrix(x.value[2, :])
        oyaw = _get_nparray_from_matrix(x.value[3, :])
        oa = _get_nparray_from_matrix(u.value[0, :])
        odelta = _get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc...", file=sys.stderr)
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def _iterative_linear_mpc_control(x0, oa, od, state, cx, cy, cyaw, dl, target_ind):
    """
    MPC contorl with updating operational point iteraitvely
    :param state:
    :param cx:
    :param cy:
    :param cyaw:
    :param dl:
    :param target_ind:
    """

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    ov = None

    for _ in range(MAX_ITER):
        xref, target_ind, dref, reaches_end = _calc_ref_trajectory(state, cx, cy, cyaw, dl, target_ind, ov)
        xbar = _predict_motion(x0, oa, od, xref)
        # poa, pod = oa, od
        oa, od, ox, oy, oyaw, ov = _linear_mpc_control(xref, xbar, x0, dref, reaches_end)
        # du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        # if du <= DU_TH:
        #     break
    # else:
    #     print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov, xref, target_ind


def _get_xy_cost_mtx_for_orientation(angle: float):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c ** 2, c * s],
        [c * s, s ** 2]
    ])


class History:

    def __init__(self):
        self.x: List[float] = []
        self.y: List[float] = []
        self.yaw: List[float] = []
        self.v: List[float] = []
        self.t: List[float] = []
        self.d: List[float] = []
        self.a: List[float] = []

    def store_state(self, state: State):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)

    def store(self, state: State, a: float, d: float):
        self.store_state(state)
        self.t.append(self.get_current_time() + DT)
        self.d.append(d)
        self.a.append(a)

    def get_current_time(self) -> float:
        return self.t[-1] if len(self.t) > 0 else 0.


def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


class MPC:
    def __init__(self, cx: np.ndarray, cy: np.ndarray, cyaw: np.ndarray, dl: float):
        """
        Simulation
        cx: course x position list
        cy: course y position list
        cyaw: course yaw position list
        dl: course tick [m]
        """

        self.cx = cx
        self.cy = cy

        cyaw = smooth_yaw(cyaw)
        self.cyaw = cyaw
        self.dl = dl

        self.goal: Tuple[float, float] = cx[-1], cy[-1]

        self.target_ind: int = 0

        self.odelta: Optional[List[float]] = None
        self.oa: Optional[List[float]] = None

        self.di: float = 0.0
        self.ai: float = 0.0

        # cyaw = smooth_yaw(cyaw)

    def step(self, state: State) -> Tuple[float, float]:
        # initial yaw compensation
        # if state.yaw - cyaw[0] >= math.pi:
        #     state.yaw -= math.pi * 2.0
        # elif state.yaw - cyaw[0] <= -math.pi:
        #     state.yaw += math.pi * 2.0

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        self.oa, self.odelta, self.ox, self.oy, self.oyaw, self.ov, self.xref, self.target_ind = \
            _iterative_linear_mpc_control(x0, self.oa, self.odelta, state, self.cx, self.cy,
                                          self.cyaw, self.dl, self.target_ind)

        if self.odelta is not None:
            self.di, self.ai = self.odelta[0], self.oa[0]
        else:
            self.ai = MAX_DECEL

        return self.di, self.ai

    def is_goal(self, state: State) -> bool:
        # check goal
        dx = state.x - self.goal[0]
        dy = state.y - self.goal[1]
        d = math.hypot(dx, dy)

        isgoal = (d <= GOAL_DIS)

        if abs(self.target_ind - len(self.cx)) >= 5:
            isgoal = False

        isstop = (abs(state.v) <= STOP_SPEED)

        if isgoal and isstop:
            return True

        return False
