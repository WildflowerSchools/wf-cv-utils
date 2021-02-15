import cv_utils.core
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

def compare_calibrations(
    old_calibrations,
    new_calibrations
):
    old_device_ids = set(old_calibrations.keys())
    new_device_ids = set(new_calibrations.keys())
    common_device_ids = old_device_ids.intersection(new_device_ids)
    logger.info('Comparing {} old calibrations with {} new calibrations'.format(
        len(old_device_ids),
        len(new_device_ids)
    ))
    logger.info('Old and new calibrations contain {} common devices'.format(
        len(common_device_ids)
    ))
    calibration_comparisons = dict()
    for device_id in common_device_ids:
        relative_rotation_vector, relative_translation_vector = cv_utils.core.compose_transformations(
            rotation_vector_1=new_calibrations[device_id]['rotation_vector'],
            translation_vector_1=np.array([0.0, 0.0, 0.0]),
            rotation_vector_2=-old_calibrations[device_id]['rotation_vector'],
            translation_vector_2=np.array([0.0, 0.0, 0.0])
        )
        angle_difference = np.linalg.norm(relative_rotation_vector)
        angle_difference_degrees = angle_difference*360/(2*math.pi)
        rotation_direction = relative_rotation_vector/angle_difference
        old_position = cv_utils.core.extract_camera_position(
            rotation_vector=old_calibrations[device_id]['rotation_vector'],
            translation_vector=old_calibrations[device_id]['translation_vector'],
        )
        new_position = cv_utils.core.extract_camera_position(
            rotation_vector=new_calibrations[device_id]['rotation_vector'],
            translation_vector=new_calibrations[device_id]['translation_vector'],
        )
        position_difference = new_position - old_position
        distance = np.linalg.norm(position_difference)
        translation_direction = position_difference/distance
        calibration_comparisons[device_id] = {
            'angle_difference': angle_difference,
            'angle_difference_degrees': angle_difference_degrees,
            'rotation_direction': rotation_direction,
            'distance': distance,
            'translation_direction': translation_direction
        }
    return calibration_comparisons
