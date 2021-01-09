import cv_datetime_utils
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import json
import os

def compose_transformations(
        rotation_vector_1,
        translation_vector_1,
        rotation_vector_2,
        translation_vector_2):
    rotation_vector_1 = np.asarray(rotation_vector_1).reshape(3)
    translation_vector_1 = np.asarray(translation_vector_1).reshape(3)
    rotation_vector_2 = np.asarray(rotation_vector_2).reshape(3)
    translation_vector_2 = np.asarray(translation_vector_2).reshape(3)
    rotation_vector_composed, translation_vector_composed = cv.composeRT(
        rotation_vector_1,
        translation_vector_1,
        rotation_vector_2,
        translation_vector_2)[:2]
    rotation_vector_composed = np.squeeze(rotation_vector_composed)
    translation_vector_composed = np.squeeze(translation_vector_composed)
    return rotation_vector_composed, translation_vector_composed


def invert_transformation(
        rotation_vector,
        translation_vector):
    rotation_vector = np.asarray(rotation_vector).reshape(3)
    translation_vector = np.asarray(translation_vector).reshape(3)
    new_rotation_vector, new_translation_vector = compose_transformations(
        np.array([0.0, 0.0, 0.0]),
        -translation_vector,
        -rotation_vector,
        np.array([0.0, 0.0, 0.0]))
    new_rotation_vector = np.squeeze(new_rotation_vector)
    new_translation_vector = np.squeeze(new_translation_vector)
    return new_rotation_vector, new_translation_vector

def quaternion_vector_to_rotation_vector(quaternion_vector):
    quaternion_vector = np.asarray(quaternion_vector).reshape(4)
    spatial_vector = quaternion_vector[1:]
    qw = quaternion_vector[0]
    spatial_vector_length = np.linalg.norm(spatial_vector)
    unit_vector = spatial_vector/spatial_vector_length
    theta = 2*np.arctan2(spatial_vector_length, qw)
    rotation_vector = theta*unit_vector
    return rotation_vector

def quaternion_vector_to_rotation_matrix(quaternion_vector):
    quaternion_tuple = tuple(np.asarray(quaternion_vector).reshape(4))
    qw, qx, qy, qz = quaternion_tuple
    R = np.array([
        [qw**2 + qx**2 - qy**2 - qz**2, 2*(qx*qy - qw*qz), 2*(qw*qy + qx*qz)],
        [2*(qx*qy + qw*qz), qw**2 - qx**2 + qy**2 - qz**2, 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qw*qx + qy*qz), qw**2 - qx**2 - qy**2 + qz**2]
    ])
    return R

def rotation_vector_to_rotation_matrix(rotation_vector):
    rotation_vector = np.asarray(rotation_vector).reshape(3)
    rotation_matrix = cv.Rodrigues(rotation_vector)[0]
    return rotation_matrix

def transform_object_points(
        object_points,
        rotation_vector=np.array([0.0, 0.0, 0.0]),
        translation_vector=np.array([0.0, 0.0, 0.0])):
    object_points = np.asarray(object_points)
    rotation_vector = np.asarray(rotation_vector)
    translation_vector = np.asarray(translation_vector)
    if object_points.size == 0:
        return object_points
    object_points = object_points.reshape((-1, 3))
    rotation_vector = rotation_vector.reshape(3)
    translation_vector = translation_vector.reshape(3)
    transformed_points = np.add(
        np.matmul(
            cv.Rodrigues(rotation_vector)[0],
            object_points.T).T,
        translation_vector.reshape((1, 3)))
    transformed_points = np.squeeze(transformed_points)
    return transformed_points


def generate_camera_pose(
        camera_position=np.array([0.0, 0.0, 0.0]),
        yaw=0.0,
        pitch=0.0,
        roll=0.0):
    # yaw: 0.0 points north (along the positive y-axis), positive angles rotate counter-clockwise
    # pitch: 0.0 is level with the ground, positive angles rotate upward
    # roll: 0.0 is level with the ground, positive angles rotate clockwise
    # All angles in radians
    camera_position = np.asarray(camera_position).reshape(3)
    # First: Move the camera to the specified position
    rotation_vector_1 = np.array([0.0, 0.0, 0.0])
    translation_vector_1 = -camera_position
    # Second: Rotate the camera so when we lower to the specified inclination, it will point in the specified compass direction
    rotation_vector_2 = np.array([0.0, 0.0, -(yaw - np.pi / 2)])
    translation_vector_2 = np.array([0.0, 0.0, 0.0])
    # Third: Lower to the specified inclination
    rotation_vector_2_3 = np.array([(np.pi / 2 - pitch), 0.0, 0.0])
    translation_vector_2_3 = np.array([0.0, 0.0, 0.0])
    # Fourth: Roll the camera by the specified angle
    rotation_vector_2_3_4 = np.array([0.0, 0.0, -roll])
    translation_vector_2_3_4 = np.array([0.0, 0.0, 0.0])
    # Combine these four moves
    rotation_vector_1_2, translation_vector_1_2 = compose_transformations(
        rotation_vector_1,
        translation_vector_1,
        rotation_vector_2,
        translation_vector_2)
    rotation_vector_1_2_3, translation_vector_1_2_3 = compose_transformations(
        rotation_vector_1_2,
        translation_vector_1_2,
        rotation_vector_2_3,
        translation_vector_2_3)
    rotation_vector, translation_vector = compose_transformations(
        rotation_vector_1_2_3,
        translation_vector_1_2_3,
        rotation_vector_2_3_4,
        translation_vector_2_3_4)
    rotation_vector = np.squeeze(rotation_vector)
    translation_vector = np.squeeze(translation_vector)
    return rotation_vector, translation_vector


def extract_camera_position(
        rotation_vector,
        translation_vector):
    rotation_vector = np.asarray(rotation_vector).reshape(3)
    translation_vector = np.asarray(translation_vector).reshape(3)
    new_rotation_vector, new_translation_vector = compose_transformations(
        rotation_vector,
        translation_vector,
        -rotation_vector,
        np.array([0.0, 0.0, 0.0]))
    camera_position = -np.squeeze(new_translation_vector)
    return camera_position


def extract_camera_direction(
        rotation_vector,
        translation_vector):
    rotation_vector = np.asarray(rotation_vector).reshape(3)
    translation_vector = np.asarray(translation_vector).reshape(3)
    camera_direction = np.matmul(
        cv.Rodrigues(-rotation_vector)[0],
        np.array([[0.0], [0.0], [1.0]]))
    camera_direction = np.squeeze(camera_direction)
    return camera_direction


def reconstruct_z_rotation(x, y):
    if x >= 0.0 and y >= 0.0:
        return np.arctan(y / x)
    if x >= 0.0 and y < 0.0:
        return np.arctan(y / x) + 2 * np.pi
    return np.arctan(y / x) + np.pi


# Currently unused; needs to be fixed up for cases in which x and/or y are close
# to zero
def extract_yaw_from_camera_direction(
        camera_direction):
    camera_direction = np.asarray(camera_direction).reshape(3)
    yaw = reconstruct_z_rotation(
        camera_direction[0],
        camera_direction[1])
    return yaw


def generate_camera_matrix(
        focal_length,
        principal_point):
    focal_length = np.asarray(focal_length).reshape(2)
    principal_point = np.asarray(principal_point).reshape(2)
    camera_matrix = np.array([
        [focal_length[0], 0, principal_point[0]],
        [0, focal_length[1], principal_point[1]],
        [0, 0, 1.0]])
    return camera_matrix


def generate_projection_matrix(
        camera_matrix,
        rotation_vector,
        translation_vector):
    camera_matrix = np.asarray(camera_matrix).reshape((3, 3))
    rotation_vector = np.asarray(rotation_vector).reshape(3)
    translation_vector = np.asarray(translation_vector).reshape(3)
    projection_matrix = np.matmul(
        camera_matrix,
        np.concatenate((
            cv.Rodrigues(rotation_vector)[0],
            translation_vector.reshape((3, 1))),
            axis=1))
    return(projection_matrix)

def ground_grid_camera_view(
    image_width,
    image_height,
    rotation_vector,
    translation_vector,
    camera_matrix,
    distortion_coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
    fill_image=False,
    step=0.1
):
    grid_corners = ground_rectangle_camera_view(
        image_width=image_width,
        image_height=image_height,
        rotation_vector=rotation_vector,
        translation_vector=translation_vector,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        fill_image=fill_image
    )
    grid_points = generate_ground_grid(
        grid_corners=grid_corners,
        step=step
    )
    return grid_points

def ground_rectangle_camera_view(
    image_width,
    image_height,
    rotation_vector,
    translation_vector,
    camera_matrix,
    distortion_coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
    fill_image=False
):
    image_points = np.array([
        [0.0, 0.0],
        [image_width, 0.0],
        [image_width, image_height],
        [0.0, image_height]
    ])
    ground_points=np.empty((4, 3))
    for i in range(4):
        ground_points[i] = ground_point(
            image_point=image_points[i],
            rotation_vector=rotation_vector,
            translation_vector=translation_vector,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients
        )
    x_values_sorted = np.sort(ground_points[:, 0])
    y_values_sorted = np.sort(ground_points[:, 1])
    if fill_image:
        x_min = x_values_sorted[0]
        x_max = x_values_sorted[3]
        y_min = y_values_sorted[0]
        y_max = y_values_sorted[3]
    else:
        x_min = x_values_sorted[1]
        x_max = x_values_sorted[2]
        y_min = y_values_sorted[1]
        y_max = y_values_sorted[2]
    return np.array([
        [x_min, y_min],
        [x_max, y_max]
    ])

def ground_point(
    image_point,
    rotation_vector,
    translation_vector,
    camera_matrix,
    distortion_coefficients=np.array([0.0, 0.0, 0.0, 0.0])
):
    image_point = np.asarray(image_point)
    rotation_vector = np.asarray(rotation_vector)
    translation_vector = np.asarray(translation_vector)
    camera_matrix = np.asarray(camera_matrix)
    distortion_coefficients = np.asarray(distortion_coefficients)
    image_point = image_point.reshape((2))
    rotation_vector = rotation_vector.reshape(3)
    translation_vector = translation_vector.reshape(3)
    camera_matrix = camera_matrix.reshape((3, 3))
    image_point_undistorted = cv.undistortPoints(
        image_point,
        camera_matrix,
        distortion_coefficients,
        P=camera_matrix
    )
    image_point_undistorted = np.squeeze(image_point_undistorted)
    camera_position = np.matmul(
        cv.Rodrigues(-rotation_vector)[0],
        -translation_vector.T
        ).T
    camera_point_homogeneous = np.matmul(
        np.linalg.inv(camera_matrix),
        np.array([image_point_undistorted[0], image_point_undistorted[1], 1.0]).T
    ).T
    camera_direction = np.matmul(
        cv.Rodrigues(-rotation_vector)[0],
        camera_point_homogeneous.T
    ).T
    theta = -camera_position[2]/camera_direction[2]
    ground_point = camera_position + theta*camera_direction
    return ground_point

def generate_ground_grid(
    grid_corners,
    step=0.1
):
    x_grid, y_grid = np.meshgrid(
    np.arange(grid_corners[0, 0], grid_corners[1, 0], step=step),
    np.arange(grid_corners[0, 1], grid_corners[1, 1], step=step)
    )
    grid = np.stack((x_grid, y_grid, np.full_like(x_grid, 0.0)), axis=-1)
    points = grid.reshape((-1, 3))
    return points

def project_points(
    object_points,
    rotation_vector,
    translation_vector,
    camera_matrix,
    distortion_coefficients,
    remove_behind_camera=False,
    remove_outside_frame=False,
    image_corners=None
):
    object_points = np.asarray(object_points).reshape((-1, 3))
    rotation_vector = np.asarray(rotation_vector).reshape(3)
    translation_vector = np.asarray(translation_vector).reshape(3)
    camera_matrix = np.asarray(camera_matrix).reshape((3, 3))
    distortion_coefficients = np.squeeze(np.asarray(distortion_coefficients))
    if object_points.size == 0:
        return np.zeros((0, 2))
    image_points = cv.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients
    )[0]
    if remove_behind_camera:
        behind_camera_boolean = behind_camera(
            object_points,
            rotation_vector,
            translation_vector
        )
        image_points[behind_camera_boolean] = np.array([np.nan, np.nan])
    if remove_outside_frame:
        outside_frame_boolean = outside_frame(
            object_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion_coefficients,
            image_corners
        )
        image_points[outside_frame_boolean] = np.array([np.nan, np.nan])
    image_points = np.squeeze(image_points)
    return image_points

def behind_camera(
        object_points,
        rotation_vector,
        translation_vector):
    object_points = np.asarray(object_points)
    rotation_vector = np.asarray(rotation_vector)
    translation_vector = np.asarray(translation_vector)
    if object_points.size == 0:
        return np.zeros((0, 2))
    object_points = object_points.reshape((-1, 3))
    rotation_vector = rotation_vector.reshape(3)
    translation_vector = translation_vector.reshape(3)
    object_points_transformed = transform_object_points(
        object_points,
        rotation_vector,
        translation_vector
    )
    behind_camera_boolean = (object_points_transformed <= 0)[..., 2]
    return behind_camera_boolean

def outside_frame(
    object_points,
    rotation_vector,
    translation_vector,
    camera_matrix,
    distortion_coefficients,
    image_corners
):
    object_points = np.asarray(object_points).reshape((-1, 3))
    rotation_vector = np.asarray(rotation_vector)
    translation_vector = np.asarray(translation_vector).reshape(3)
    camera_matrix = np.asarray(camera_matrix).reshape((3,3))
    distortion_coefficients = np.squeeze(np.asarray(distortion_coefficients))
    image_corners = np.asarray(image_corners).reshape((2,2))
    if object_points.size == 0:
        return np.zeros((0, 2))
    image_points = cv.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        np.array([0.0, 0.0, 0.0, 0.0])
    )[0]
    image_points = image_points.reshape((-1, 2))
    outside_frame_boolean = (
        (image_points[:, 0] < image_corners[0, 0]) |
        (image_points[:, 0] > image_corners[1, 0]) |
        (image_points[:, 1] < image_corners[0, 1]) |
        (image_points[:, 1] > image_corners[1, 1])
    )
    return outside_frame_boolean


def undistort_points(
        image_points,
        camera_matrix,
        distortion_coefficients):
    image_points = np.asarray(image_points)
    camera_matrix = np.asarray(camera_matrix)
    distortion_coefficients = np.asarray(distortion_coefficients)
    if image_points.size == 0:
        return image_points
    image_points = image_points.reshape((-1, 1, 2))
    camera_matrix = camera_matrix.reshape((3, 3))
    undistorted_points = cv.undistortPoints(
        image_points,
        camera_matrix,
        distortion_coefficients,
        P=camera_matrix)
    undistorted_points = np.squeeze(undistorted_points)
    return undistorted_points


def estimate_camera_pose_from_image_points(
        image_points_1,
        image_points_2,
        camera_matrix,
        rotation_vector_1=np.array([0.0, 0.0, 0.0]),
        translation_vector_1=np.array([0.0, 0.0, 0.0]),
        distance_between_cameras=1.0):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix = np.asarray(camera_matrix)
    rotation_vector_1 = np.asarray(rotation_vector_1)
    translation_vector_1 = np.asarray(translation_vector_1)
    if image_points_1.size == 0 or image_points_2.size == 0:
        raise ValueError('One or both sets of image points appear to be empty')
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    camera_matrix = camera_matrix.reshape((3, 3))
    rotation_vector_1 = rotation_vector_1.reshape(3)
    translation_vector_1 = translation_vector_1.reshape(3)
    essential_matrix, mask = cv.findEssentialMat(
        image_points_1,
        image_points_2,
        camera_matrix)
    relative_rotation_matrix, relative_translation_vector = cv.recoverPose(
        essential_matrix,
        image_points_1,
        image_points_2,
        camera_matrix,
        mask=mask)[1:3]
    relative_rotation_vector = cv.Rodrigues(relative_rotation_matrix)[0]
    relative_translation_vector = relative_translation_vector * distance_between_cameras
    rotation_vector_2, translation_vector_2 = compose_transformations(
        rotation_vector_1,
        translation_vector_1,
        relative_rotation_vector,
        relative_translation_vector)
    rotation_vector_2 = np.squeeze(rotation_vector_2)
    translation_vector_2 = np.squeeze(translation_vector_2)
    return rotation_vector_2, translation_vector_2


def reconstruct_object_points_from_camera_poses(
        image_points_1,
        image_points_2,
        camera_matrix,
        rotation_vector_1,
        translation_vector_1,
        rotation_vector_2,
        translation_vector_2):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix = np.asarray(camera_matrix)
    rotation_vector_1 = np.asarray(rotation_vector_1)
    translation_vector_1 = np.asarray(translation_vector_1)
    rotation_vector_2 = np.asarray(rotation_vector_2)
    translation_vector_2 = np.asarray(translation_vector_2)
    if image_points_1.size == 0 or image_points_2.size == 0:
        return np.zeros((0, 3))
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    camera_matrix = camera_matrix.reshape((3, 3))
    rotation_vector_1 = rotation_vector_1.reshape(3)
    translation_vector_1 = translation_vector_1.reshape(3)
    rotation_vector_2 = rotation_vector_2.reshape(3)
    translation_vector_2 = translation_vector_2.reshape(3)
    projection_matrix_1 = generate_projection_matrix(
        camera_matrix,
        rotation_vector_1,
        translation_vector_1)
    projection_matrix_2 = generate_projection_matrix(
        camera_matrix,
        rotation_vector_2,
        translation_vector_2)
    object_points_homogeneous = cv.triangulatePoints(
        projection_matrix_1,
        projection_matrix_2,
        image_points_1.T,
        image_points_2.T)
    object_points = cv.convertPointsFromHomogeneous(
        object_points_homogeneous.T)
    object_points = np.squeeze(object_points)
    return object_points


def reconstruct_object_points_from_relative_camera_pose(
        image_points_1,
        image_points_2,
        camera_matrix,
        relative_rotation_vector,
        relative_translation_vector,
        rotation_vector_1=np.array([[0.0], [0.0], [0.0]]),
        translation_vector_1=np.array([[0.0], [0.0], [0.0]]),
        distance_between_cameras=1.0):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix = np.asarray(camera_matrix)
    relative_rotation_vector = np.asarray(relative_rotation_vector)
    relative_translation_vector = np.asarray(relative_translation_vector)
    rotation_vector_1 = np.asarray(rotation_vector_1)
    translation_vector_1 = np.asarray(translation_vector_1)
    if image_points_1.size == 0 or image_points_2.size == 0:
        return np.zeros((0, 3))
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    camera_matrix = camera_matrix.reshape((3, 3))
    relative_rotation_vector = relative_rotation_vector.reshape(3)
    relative_translation_vector = relative_translation_vector.reshape(3)
    rotation_vector_1 = rotation_vector_1.reshape(3)
    translation_vector_1 = translation_vector_1.reshape(3)
    rotation_vector_2, translation_vector_2 = cv.composeRT(
        rotation_vector_1,
        translation_vector_1,
        relative_rotation_vector,
        relative_translation_vector * distance_between_cameras)[:2]
    object_points = reconstruct_object_points_from_camera_poses(
        image_points_1,
        image_points_2,
        camera_matrix,
        rotation_vector_1,
        translation_vector_1,
        rotation_vector_2,
        translation_vector_2)
    return object_points


def reconstruct_object_points_from_image_points(
        image_points_1,
        image_points_2,
        camera_matrix,
        rotation_vector_1=np.array([[0.0], [0.0], [0.0]]),
        translation_vector_1=np.array([[0.0], [0.0], [0.0]]),
        distance_between_cameras=1.0):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix = np.asarray(camera_matrix)
    rotation_vector_1 = np.asarray(rotation_vector_1)
    translation_vector_1 = np.asarray(translation_vector_1)
    if image_points_1.size == 0 or image_points_2.size == 0:
        return np.zeros((0, 3))
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    camera_matrix = camera_matrix.reshape((3, 3))
    rotation_vector_1 = rotation_vector_1.reshape(3)
    translation_vector_1 = translation_vector_1.reshape(3)
    rotation_vector_2, translation_vector_2 = estimate_camera_pose_from_image_points(
        image_points_1,
        image_points_2,
        camera_matrix,
        rotation_vector_1,
        translation_vector_1,
        distance_between_cameras)
    object_points = reconstruct_object_points_from_camera_poses(
        image_points_1,
        image_points_2,
        camera_matrix,
        rotation_vector_1,
        translation_vector_1,
        rotation_vector_2,
        translation_vector_2)
    return object_points


def estimate_camera_pose_from_plane_object_points(
        input_object_points,
        height,
        origin_index,
        x_axis_index,
        y_reference_point,
        y_reference_point_sign,
        distance_calibration_indices,
        calibration_distance):
    input_object_points = np.asarray(input_object_points)
    if input_object_points.size == 0:
        raise ValueError('Obect point array appears to be empty')
    input_object_points = input_object_points.reshape((-1, 3))

    scale_factor = np.divide(
        calibration_distance,
        np.linalg.norm(
            np.subtract(
                input_object_points[distance_calibration_indices[0]],
                input_object_points[distance_calibration_indices[1]])))

    object_points_1 = np.multiply(
        input_object_points,
        scale_factor)

    def objective_function(parameters):
        rotation_x = parameters[0]
        rotation_y = parameters[1]
        translation_z = parameters[2]
        object_points_transformed = transform_object_points(
            object_points_1,
            np.array([rotation_x, rotation_y, 0.0]),
            np.array([0.0, 0.0, translation_z]))
        return np.sum(np.square(object_points_transformed[:, 2] - height))

    optimization_solution = scipy.optimize.minimize(
        objective_function,
        np.array([0.0, 0.0, 0.0]))

    rotation_x_a = optimization_solution['x'][0]
    rotation_y_a = optimization_solution['x'][1]
    translation_z_a = optimization_solution['x'][2]

    rotation_x_rotation_y_a_norm = np.linalg.norm([rotation_x_a, rotation_y_a])

    rotation_x_b = rotation_x_a * ((rotation_x_rotation_y_a_norm + np.pi) / rotation_x_rotation_y_a_norm)
    rotation_y_b = rotation_y_a * ((rotation_x_rotation_y_a_norm + np.pi) / rotation_x_rotation_y_a_norm)
    translation_z_b = - translation_z_a

    rotation_vector_2_a = np.array([rotation_x_a, rotation_y_a, 0.0])
    translation_vector_2_a = np.array([0.0, 0.0, translation_z_a])
    object_points_2_a = transform_object_points(
        object_points_1,
        rotation_vector_2_a,
        translation_vector_2_a)

    rotation_vector_2_b = np.array([rotation_x_b, rotation_y_b, 0.0])
    translation_vector_2_b = np.array([0.0, 0.0, translation_z_b])
    object_points_2_b = transform_object_points(
        object_points_1,
        rotation_vector_2_b,
        translation_vector_2_b)

    sign_a = np.sign(
        np.cross(
            np.subtract(
                object_points_2_a[x_axis_index],
                object_points_2_a[origin_index]),
            np.subtract(
                object_points_2_a[y_reference_point],
                object_points_2_a[origin_index]))[2])

    sign_b = np.sign(
        np.cross(
            np.subtract(
                object_points_2_b[x_axis_index],
                object_points_2_b[origin_index]),
            np.subtract(
                object_points_2_b[y_reference_point],
                object_points_2_b[origin_index]))[2])

    if sign_a == y_reference_point_sign:
        rotation_vector_2 = rotation_vector_2_a
        translation_vector_2 = translation_vector_2_a
        object_points_2 = object_points_2_a
    else:
        rotation_vector_2 = rotation_vector_2_b
        translation_vector_2 = translation_vector_2_b
        object_points_2 = object_points_2_b

    xy_shift = - object_points_2[origin_index, :2]

    rotation_vector_3 = np.array([0.0, 0.0, 0.0])
    translation_vector_3 = np.array([xy_shift[0], xy_shift[1], 0.0])
    object_points_3 = transform_object_points(
        object_points_2,
        rotation_vector_3,
        translation_vector_3)

    final_z_rotation = - reconstruct_z_rotation(
        object_points_3[x_axis_index, 0],
        object_points_3[x_axis_index, 1])

    rotation_vector_4 = np.array([0.0, 0.0, final_z_rotation])
    translation_vector_4 = np.array([0.0, 0.0, 0.0])
    object_points_4 = transform_object_points(
        object_points_3,
        rotation_vector_4,
        translation_vector_4)

    rotation_vector_2_3, translation_vector_2_3 = compose_transformations(
        rotation_vector_2,
        translation_vector_2,
        rotation_vector_3,
        translation_vector_3)

    rotation_vector_2_3_4, translation_vector_2_3_4 = compose_transformations(
        rotation_vector_2_3,
        translation_vector_2_3,
        rotation_vector_4,
        translation_vector_4)

    camera_rotation_vector, camera_translation_vector = invert_transformation(
        rotation_vector_2_3_4,
        translation_vector_2_3_4)

    return camera_rotation_vector, camera_translation_vector, scale_factor, object_points_4


def estimate_camera_poses_from_plane_image_points(
        image_points_1,
        image_points_2,
        camera_matrix,
        height,
        origin_index,
        x_axis_index,
        y_reference_point,
        y_reference_point_sign,
        distance_calibration_indices,
        calibration_distance):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix = np.asarray(camera_matrix)
    if image_points_1.size == 0 or image_points_2.size == 0:
        raise ValueError('One or both sets of image points appear to be empty')
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    camera_matrix = camera_matrix.reshape((3, 3))
    relative_rotation_vector, relative_translation_vector = estimate_camera_pose_from_image_points(
        image_points_1,
        image_points_2,
        camera_matrix)
    input_object_points = reconstruct_object_points_from_image_points(
        image_points_1,
        image_points_2,
        camera_matrix)
    rotation_vector_1, translation_vector_1, scale_factor = estimate_camera_pose_from_plane_object_points(
        input_object_points,
        height,
        origin_index,
        x_axis_index,
        y_reference_point,
        y_reference_point_sign,
        distance_calibration_indices,
        calibration_distance)[:3]
    rotation_vector_2, translation_vector_2 = compose_transformations(
        rotation_vector_1,
        translation_vector_1,
        relative_rotation_vector,
        relative_translation_vector * scale_factor)
    return rotation_vector_1, translation_vector_1, rotation_vector_2, translation_vector_2
