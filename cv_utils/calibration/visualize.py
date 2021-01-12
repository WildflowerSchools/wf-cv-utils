import cv_utils.core
import cv_utils.drawing.opencv
import numpy as np
import math

def generate_grid_corner_object_points(
    room_corners,
    floor_height=0.0
):
    grid_corners = generate_grid_corners(room_corners)
    grid_corner_object_points=np.array([
        [grid_corners[0, 0], grid_corners[0, 1], floor_height],
        [grid_corners[0, 0], grid_corners[1, 1], floor_height],
        [grid_corners[1, 0], grid_corners[0, 1], floor_height],
        [grid_corners[1, 0], grid_corners[1, 1], floor_height]
    ])
    return grid_corner_object_points

def generate_floor_grid_object_points(
    room_corners,
    floor_height=0.0,
    num_points_per_distance_unit=2
):
    num_points_per_distance_unit = round(num_points_per_distance_unit)
    grid_corners = generate_grid_corners(room_corners)
    x_grid, y_grid = np.meshgrid(
        np.linspace(
            grid_corners[0, 0],
            grid_corners[1, 0],
            num=round(grid_corners[1, 0] - grid_corners[0, 0])*num_points_per_distance_unit + 1,
            endpoint=True
        ),
        np.linspace(
            grid_corners[0, 1],
            grid_corners[1, 1],
            num=round(grid_corners[1, 1] - grid_corners[0, 1])*num_points_per_distance_unit + 1,
            endpoint=True
            )
    )
    grid = np.stack((x_grid, y_grid, np.full_like(x_grid, floor_height)), axis=-1)
    object_points = grid.reshape((-1, 3))
    return object_points

def generate_grid_corners(room_corners):
    room_corners = np.asarray(room_corners)
    grid_corners = np.array([
        [float(math.ceil(room_corners[0, 0])), float(math.ceil(room_corners[0, 1]))],
        [float(math.floor(room_corners[1, 0])), float(math.floor(room_corners[1, 1]))],
    ])
    return grid_corners
