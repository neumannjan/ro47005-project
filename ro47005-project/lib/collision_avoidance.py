import math
from typing import List, Tuple, Union, Optional

import numpy as np

from lib.car_dimensions import CarDimensions
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


def _find_intersection_points_nonparallel(traj1: np.ndarray, traj2: np.ndarray) -> Optional[
    Tuple[np.ndarray, np.ndarray]]:
    if len(traj1) <= 1 or len(traj2) <= 1:
        return None

    A = traj1[:-1, :2]
    B = traj1[1:, :2]
    C = traj2[:-1, :2]
    D = traj2[1:, :2]

    A = np.repeat(A, axis=0, repeats=len(traj2) - 1)
    B = np.repeat(B, axis=0, repeats=len(traj2) - 1)

    C = np.concatenate([C] * (len(traj1) - 1), axis=0)
    D = np.concatenate([D] * (len(traj1) - 1), axis=0)

    s1 = B - A
    s2 = D - C

    denom = -s2[:, 0] * s1[:, 1] + s1[:, 0] * s2[:, 1]
    s = (-s1[:, 1] * (A[:, 0] - C[:, 0]) + s1[:, 0] * (A[:, 1] - C[:, 1])) / denom
    t = (s2[:, 0] * (A[:, 1] - C[:, 1]) - s2[:, 1] * (A[:, 0] - C[:, 0])) / denom

    mask_intersect = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)

    idx_intersect, = np.where(mask_intersect)
    idx_traj1_intersect = idx_intersect // (len(traj2) - 1)

    if len(idx_intersect) == 0:
        return None

    pts = A[mask_intersect] + t[mask_intersect] * s1[mask_intersect]
    return pts, idx_traj1_intersect


def _fix_idx_by_mask(idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    idx_all, = np.where(mask)
    return idx_all[idx]


def _find_intersection_points_parallel(traj1: np.ndarray, traj2: np.ndarray, car_dimensions: CarDimensions) -> Optional[
    Tuple[np.ndarray, np.ndarray]]:
    # check if they are sufficiently close
    diffs = traj1[:, :2] - traj2[:, :2]
    angles = traj1[:, 2] + np.pi / 2
    is_close = np.sqrt((np.cos(angles) * diffs[:, 0] + np.sin(angles) * diffs[:, 1]) ** 2) <= 2 * car_dimensions.radius

    idx, = np.where(is_close)
    pts = traj1[is_close, :2]

    return pts, idx


def _determine_parallel(traj1: np.ndarray, traj2: np.ndarray) -> np.ndarray:
    yaw_diff = np.abs(traj1[:, 2] - traj2[:, 2]) % math.tau
    is_parallel = yaw_diff <= math.pi / 8
    return is_parallel


def _find_intersection_points(traj1: np.ndarray, traj2: np.ndarray,
                              car_dimensions: CarDimensions) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    is_parallel = _determine_parallel(traj1, traj2)

    result1 = _find_intersection_points_nonparallel(traj1[~is_parallel], traj2[~is_parallel])
    result2 = _find_intersection_points_parallel(traj1[is_parallel], traj2[is_parallel], car_dimensions)

    if result1 is not None and result2 is not None:
        pts1, idx1 = result1
        idx1 = _fix_idx_by_mask(idx1, ~is_parallel)
        pts2, idx2 = result2
        idx2 = _fix_idx_by_mask(idx2, is_parallel)
        return np.concatenate([pts1, pts2]), np.concatenate([idx1, idx2])

    if result1 is not None:
        pts1, idx1 = result1
        idx1 = _fix_idx_by_mask(idx1, ~is_parallel)
        return pts1, idx1

    if result2 is not None:
        pts2, idx2 = result2
        idx2 = _fix_idx_by_mask(idx2, is_parallel)
        return pts2, idx2

    return None


def _get_point_collide_mask(collision_point_trajectories: List[np.ndarray], point: np.ndarray,
                            car_dimensions: CarDimensions, extra_margin: float):
    diffs = [tr[:, :2] - point[None, :2] for tr in collision_point_trajectories]
    dists = np.linalg.norm(diffs, axis=2)
    mask = np.any(dists <= 2 * car_dimensions.radius + extra_margin, axis=0)  # whether point within radius
    return mask


def _find_first_intersection_point_with_time(traj1: np.ndarray, traj2: np.ndarray, cc_traj1: List[np.ndarray],
                                             cc_traj2: List[np.ndarray], car_dimensions: CarDimensions,
                                             frame_window_l: List[int], extra_margin: float) -> Optional[
    Tuple[Tuple[float, float], int]]:
    result = _find_intersection_points(traj1, traj2, car_dimensions)

    if result is None:
        return None

    intersection_pts, idx_pts = result

    for point, idx in zip(intersection_pts, idx_pts):
        mask1 = _get_point_collide_mask(cc_traj1, point, car_dimensions, extra_margin)
        mask2 = _get_point_collide_mask(cc_traj2, point, car_dimensions, extra_margin)

        for offset in frame_window_l:
            if offset > 0:
                mask2[offset:] |= mask2[:-offset]
            elif offset < 0:
                mask2[:offset] |= mask2[-offset:]

        collides_in_time = np.any(mask1 & mask2)
        if collides_in_time:
            return tuple(point.tolist()), idx

    return None


def check_collision_moving_cars(traj_agent: np.ndarray,
                                traj_obstacles: List[np.ndarray], car_dimensions: CarDimensions, frame_window: int,
                                extra_margin: float) -> \
        Optional[
            Tuple[float, float]]:
    traj_agent, traj_obstacles = _pad_trajectories(traj_agent, traj_obstacles)

    frame_window_l = list(range(-frame_window, frame_window + 1))

    cc_traj_agent = car_trajectory_to_collision_point_trajectories(traj_agent, car_dimensions=car_dimensions)
    cc_traj_obstacles = [car_trajectory_to_collision_point_trajectories(tr, car_dimensions=car_dimensions)
                         for tr in traj_obstacles]

    min_idx = float('inf')
    min_pt = None
    for traj_obstacle, cc_traj_obstacle in zip(traj_obstacles, cc_traj_obstacles):
        intersection_pt = _find_first_intersection_point_with_time(traj_agent, traj_obstacle, cc_traj_agent,
                                                                   cc_traj_obstacle,
                                                                   car_dimensions, frame_window_l, extra_margin)

        if intersection_pt is not None:
            pt, idx = intersection_pt

            if idx < min_idx:
                min_idx = idx
                min_pt = pt

    return min_pt


def get_cutoff_curve_by_position_idx(points: np.ndarray, x: float, y: float, radius: float) -> int:
    points_diff = points[:, :2].copy()
    points_diff[:, 0] -= x
    points_diff[:, 1] -= y

    points_dist = np.linalg.norm(points_diff, axis=1) <= radius
    first_idx = np.argmax(points_dist)

    if not points_dist[first_idx]:
        # no cutoff
        return len(points)

    return int(first_idx)
