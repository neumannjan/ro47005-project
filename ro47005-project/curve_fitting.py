import json
from pathlib import Path

import numpy as np


def load_data(filename: str):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def get_approximation_functions(points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(points).T  # as column vector
    return np.concatenate([points ** 2, points], axis=1)


FILENAMES = [
    Path('./data/motion_primitives/1.0_1.0_0.2.json'),
    Path('./data/motion_primitives/1.0_1.0_0.4.json'),
    Path('./data/motion_primitives/1.0_1.0_0.6.json'),
    Path('./data/motion_primitives/1.0_1.0_0.95.json'),
]

for filename in FILENAMES:
    data = load_data(filename)
    points = np.array(data['points'])

    points_x = points[:, 0]
    points_y = points[:, 1]
    points_angle = points[:, 2]

    A = get_approximation_functions(points_x)

    # compute the optimal parameters of the resulting curves
    p_y = np.linalg.inv(A.T @ A) @ (A.T @ np.atleast_2d(points_y).T)
    p_angle = np.linalg.inv(A.T @ A) @ (A.T @ np.atleast_2d(points_angle).T)

    # select new X points
    new_points_x = np.linspace(start=points[0, 0], stop=points[-1, 0], num=points.shape[0])
    new_A = get_approximation_functions(new_points_x)

    # compute new Y points
    new_points_y = new_A @ p_y
    new_points_angle = new_A @ p_angle

    new_points = np.hstack([
        np.atleast_2d(new_points_x).T,
        new_points_y,
        new_points_angle
    ])

    data['points'] = new_points.tolist()

    with open(filename.with_name(filename.stem + "_curvefitted" + filename.suffix), 'w') as file:
        json.dump(data, file, indent=4)
