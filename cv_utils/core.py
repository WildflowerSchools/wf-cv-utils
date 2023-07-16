import cv2 as cv
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def termination_criteria(
    max_iterations=10000,
    accuracy=1e-9
):
    termination_criteria = (cv.TERM_CRITERIA_COUNT & cv.TERM_CRITERIA_EPS, max_iterations, accuracy)
    return termination_criteria

def camera_calibration_flags(
    use_intrinsic_guess=False,
    fix_principal_point=False,
    fix_aspect_ratio=False,
    zero_tangent_distortion=False,
    fix_focal_length=False,
    fix_k1=False,
    fix_k2=False,
    fix_k3=False,
    fix_k4=False,
    fix_k5=False,
    fix_k6=False,
    rational_model=False,
    thin_prism_model=False,
    fix_s1_s2_s3_s4=False,
    tilted_model=False,
    fix_taux_tauy=False,
):
    flags = 0
    flag_descriptions=list()
    if use_intrinsic_guess:
        flags += cv.CALIB_USE_INTRINSIC_GUESS
        flag_descriptions.append('use intrinsic guess')
    if fix_principal_point:
        flags += cv.CALIB_FIX_PRINCIPAL_POINT
        flag_descriptions.append('fix principal point')
    if fix_aspect_ratio:
        flags += cv.CALIB_FIX_ASPECT_RATIO
        flag_descriptions.append('fix aspect ratio')
    if zero_tangent_distortion:
        flags += cv.CALIB_ZERO_TANGENT_DIST
        flag_descriptions.append('zero tangent distortion')
    if fix_focal_length:
        flags += cv.CALIB_FIX_FOCAL_LENGTH
        flag_descriptions.append('fix focal length')
    if fix_k1:
        flags += cv.CALIB_FIX_K1
        flag_descriptions.append('fix k_1')
    if fix_k2:
        flags += cv.CALIB_FIX_K2
        flag_descriptions.append('fix k_2')
    if fix_k3:
        flags += cv.CALIB_FIX_K3
        flag_descriptions.append('fix k_3')
    if fix_k4:
        flags += cv.CALIB_FIX_K4
        flag_descriptions.append('fix k_4')
    if fix_k5:
        flags += cv.CALIB_FIX_K5
        flag_descriptions.append('fix k_5')
    if fix_k6:
        flags += cv.CALIB_FIX_K6
        flag_descriptions.append('fix k_6')
    if rational_model:
        flags += cv.CALIB_RATIONAL_MODEL
        flag_descriptions.append('rational model')
    if thin_prism_model:
        flags += cv.CALIB_THIN_PRISM_MODEL
        flag_descriptions.append('thin prism model')
    if fix_s1_s2_s3_s4:
        flags += cv.CALIB_FIX_S1_S2_S3_S4
        flag_descriptions.append('fix s_1/s_2/s_3/s_4')
    if tilted_model:
        flags += cv.CALIB_TILTED_MODEL
        flag_descriptions.append('tilted model')
    if fix_taux_tauy:
        flags += cv.CALIB_FIX_TAUX_TAUY
        flag_descriptions.append('fix tau_x/tau_y')
    return flags, flag_descriptions

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

def extract_camera_position_rotation_matrix(rotation_matrix, translation_vector):
    rotation_matrix = np.asarray(rotation_matrix).reshape((3,3))
    translation_vector = np.asarray(translation_vector).reshape(3)
    position = np.matmul(rotation_matrix.T, -translation_vector.T)
    return position

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

def write_image(
    image,
    path
):
    image_output_directory, image_output_filename = os.path.split(path)
    if image_output_directory is not None:
        os.makedirs(image_output_directory, exist_ok=True)
    cv.imwrite(
        filename=path,
        img=image
    )

def read_image(path):
    image = cv.imread(path)
    return image
