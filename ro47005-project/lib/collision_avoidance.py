from typing import List, Tuple, Union, Optional

import numpy as np

from lib.car_dimensions import CarDimensions
from lib.helpers import measure_time
from lib.trajectories import car_trajectory_to_collision_point_trajectories


def _pad_trajectory(traj: np.ndarray, n_iterations: int) -> np.ndarray:
    if len(traj) < n_iterations:
        return np.vstack([traj, np.repeat(traj[-1:, :], n_iterations - len(traj), axis=0)])
    else:
        return traj[:n_iterations]


def _pad_trajectories(traj_agent: np.ndarray, trajs_o: List[np.ndarray]):
    n_iterations = max(len(traj_agent), max((len(tr) for tr in trajs_o)))
    traj_agent = _pad_trajectory(traj_agent, n_iterations)
    trajs_o = [_pad_trajectory(tr, n_iterations) for tr in trajs_o]
    return traj_agent, trajs_o


def _find_intersections(traj1: np.ndarray, traj2: np.ndarray, diff_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    # this function is right now pretty dumb
    # computes distances between each point on traj1 with each point on traj2

    mask = np.linalg.norm(traj1[None, :, :2] - traj2[:, None, :2], axis=2) <= diff_threshold

    mask1 = np.any(mask, axis=0)
    mask2 = np.any(mask, axis=1)

    return mask1, mask2


def _get_first_idx(mask: np.ndarray):
    idx = np.argmax(mask)
    if not mask[idx]:
        return None
    return idx


def _find_first_intersection_idx(traj1: np.ndarray, traj2: np.ndarray, diff_threshold: float):
    mask1, _ = _find_intersections(traj1, traj2, diff_threshold)
    return _get_first_idx(mask1)


def check_collision_moving_cars(traj_agent: np.ndarray,
                                traj_obstacles: List[np.ndarray], diff_threshold: float) -> Optional[
    Tuple[float, float]]:

    traj_agent, traj_obstacles = _pad_trajectories(traj_agent, traj_obstacles)

    min_idx = None
    for traj_obstacle in traj_obstacles:
        idx = _find_first_intersection_idx(traj_agent, traj_obstacle, diff_threshold)
        if idx is not None and (min_idx is None or idx < min_idx):
            min_idx = idx

    if min_idx is None:
        return None

    x, y = traj_agent[min_idx, :2]
    return x, y


def get_cutoff_curve_by_position_idx(points: np.ndarray, x: float, y: float, radius: float = 0.001) -> int:
    points_diff = points[:, :2].copy()
    points_diff[:, 0] -= x
    points_diff[:, 1] -= y

    points_dist = np.linalg.norm(points_diff, axis=1) <= radius
    first_idx = np.argmax(points_dist)

    if not points_dist[first_idx]:
        # no cutoff
        return points

    return first_idx
