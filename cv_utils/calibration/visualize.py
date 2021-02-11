import cv_utils.core
import cv_utils.drawing.opencv
import minimal_honeycomb
import video_io
import cv2
import numpy as np
import datetime
import math
import os
import logging

logger = logging.getLogger(__name__)

def visualize_calibration(
    visualization_datetime,
    room_corners,
    floor_height=0.0,
    environment_id=None,
    environment_name=None,
    mark_device_locations=False,
    marked_device_types = ['UWBANCHOR', 'PI3WITHCAMERA', 'PI4WITHCAMERA', 'PIZEROWITHCAMERA'],
    num_grid_points_per_distance_unit=2,
    grid_point_radius=3,
    grid_point_color='#00ff00',
    grid_point_alpha=1.0,
    corner_label_horizontal_alignment='left',
    corner_label_vertical_alignment='bottom',
    corner_label_font_scale=1.0,
    corner_label_line_width=2,
    corner_label_color='#00ff00',
    corner_label_alpha=1.0,
    device_point_radius=3,
    device_point_color='#ff0000',
    device_point_alpha=1.0,
    device_label_horizontal_alignment='left',
    device_label_vertical_alignment='bottom',
    device_label_font_scale=1.0,
    device_label_line_width=2,
    device_label_color='#ff0000',
    device_label_alpha=1.0,
    show=False,
    save=True,
    output_directory='./overlays',
    output_filename_extension='jpg'
):
    logger.info('Launching Honeycomb client')
    client = minimal_honeycomb.MinimalHoneycombClient()
    if environment_id is None:
        if environment_name is None:
            raise ValueError('Must specify either environment ID or environment name')
        logger.info('Environment ID not specified. Fetching environmnt ID for environment name {}'.format(environment_name))
        result = client.bulk_query(
            request_name='findEnvironments',
            arguments={
                'name': {
                    'type': 'String',
                    'value': environment_name
                }
            },
            return_data=[
                'environment_id'
            ],
            id_field_name='environment_id'
        )
        if len(result) == 0:
            raise ValueError('Environment {} not found'.format(environment_name))
        if len(result) > 1:
            raise ValueError('More than one environment found with name {}'.format(environment_name))
        environment_id = result[0].get('environment_id')
    logger.info('Visualizing calibration for environment id {}'.format(environment_id))
    logger.info('Generating object points for grid')
    floor_grid_object_points = cv_utils.generate_floor_grid_object_points(
        room_corners=room_corners,
        floor_height=0.0,
        num_points_per_distance_unit=num_grid_points_per_distance_unit
    )
    grid_corner_object_points = cv_utils.generate_grid_corner_object_points(
        room_corners,
        floor_height
    )
    if mark_device_locations:
        logger.info('Fetching device assignments to mark their locations')
        result=client.bulk_query(
            request_name='searchAssignments',
            arguments={
                'query': {
                    'type': 'QueryExpression!',
                    'value': {
                        'operator': 'AND',
                        'children': [
                            {
                                'field': 'environment',
                                'operator': 'EQ',
                                'value': environment_id
                            },
                            {
                                'field': 'assigned_type',
                                'operator': 'EQ',
                                'value': 'DEVICE'
                            }
                        ]
                    }
                }
            },
            return_data=[
                'assignment_id',
                'start',
                'end',
                {'assigned': [
                    {'... on Device': [
                        'device_id',
                        'name',
                        'device_type',
                        {'position_assignments': [
                            'start',
                            'end',
                            {'coordinate_space': [
                                'space_id'
                            ]},
                            'coordinates'
                        ]}
                    ]}
                ]}
            ],
            id_field_name='assignment_id'
        )
        logger.info('Fetched {} device assignments'.format(len(result)))
        device_assignments = minimal_honeycomb.filter_assignments(
            result,
            start_time=visualization_datetime,
            end_time=visualization_datetime
        )
        logger.info('{} of these device assignments are active at specified datetime'.format(len(device_assignments)))
        device_assignments = list(filter(lambda x: x.get('assigned', {}).get('device_type') in marked_device_types, result))
        logger.info('{} of these device assignments correspond to target device types'.format(len(device_assignments)))
        device_names = list()
        device_object_points = list()
        for device_assignment in device_assignments:
            device_id = device_assignment.get('assigned', {}).get('device_id')
            device_name = device_assignment.get('assigned', {}).get('name')
            if device_name is None:
                logger.info('Device {} has no name. Skipping.'.format(device_id))
                continue
            position_assignments = device_assignment.get('assigned', {}).get('position_assignments')
            if position_assignments is None:
                continue
            logger.info('Device {} has {} position assignments'.format(device_name, len(position_assignments)))
            position_assignments = minimal_honeycomb.filter_assignments(
                position_assignments,
                start_time=visualization_datetime,
                end_time=visualization_datetime
            )
            logger.info('{} of these position assignments are active at specified datetime'.format(len(position_assignments)))
            for position_assignment in position_assignments:
                device_names.append(device_name)
                device_object_points.append(position_assignment.get('coordinates'))
        device_object_points = np.asarray(device_object_points)
        num_marked_devices = len(device_names)
        if device_object_points.shape[0] != num_marked_devices:
            raise ValueError('Found {} valid position assignments but resulting device object points has shape{}'.format(
                num_marked_devices,
                device_object_points.shape
            ))
        logger.info('Fetched {} valid position assignments'.format(len(device_names)))
    logger.info('Fetching images')
    metadata = video_io.fetch_images(
        image_timestamps=[visualization_datetime],
        environment_id=environment_id
    )
    logger.info('Fetched {} images'.format(len(metadata)))
    for metadatum in metadata:
        camera_device_id = metadatum.get('device_id')
        logger.info('Visualizing calibration for camera device id {}'.format(camera_device_id))
        logger.info('Fetching calibrating data')
        result = client.bulk_query(
            request_name='findIntrinsicCalibrations',
            arguments={
                'device': {
                    'type': 'ID',
                    'value': camera_device_id
                }
            },
            return_data = [
                'intrinsic_calibration_id',
                'start',
                'end',
                'camera_matrix',
                'distortion_coefficients',
                'image_width',
                'image_height'
            ],
            id_field_name='intrinsic_calibration_id'
        )
        intrinsic_calibration = minimal_honeycomb.extract_assignment(
            assignments=result,
            start_time=visualization_datetime,
            end_time=visualization_datetime
        )
        result = client.bulk_query(
            request_name='findExtrinsicCalibrations',
            arguments={
                'device': {
                    'type': 'ID',
                    'value': camera_device_id
                }
            },
            return_data = [
                'extrinsic_calibration_id',
                'start',
                'end',
                'rotation_vector',
                'translation_vector'
            ],
            id_field_name='extrinsic_calibration_id'
        )
        extrinsic_calibration = minimal_honeycomb.extract_assignment(
            assignments=result,
            start_time=visualization_datetime,
            end_time=visualization_datetime
        )
        logger.info('Calculating image points from object points')
        image_corners = np.array([
            [0.0, 0.0],
            [float(intrinsic_calibration.get('image_width')), float(intrinsic_calibration.get('image_height'))]
        ])
        floor_grid_image_points = cv_utils.core.project_points(
            object_points=floor_grid_object_points,
            rotation_vector=extrinsic_calibration.get('rotation_vector'),
            translation_vector=extrinsic_calibration.get('translation_vector'),
            camera_matrix=intrinsic_calibration.get('camera_matrix'),
            distortion_coefficients=intrinsic_calibration.get('distortion_coefficients'),
            remove_behind_camera=True,
            remove_outside_frame=True,
            image_corners=image_corners
        )
        grid_corner_image_points = cv_utils.core.project_points(
            object_points=grid_corner_object_points,
            rotation_vector=extrinsic_calibration.get('rotation_vector'),
            translation_vector=extrinsic_calibration.get('translation_vector'),
            camera_matrix=intrinsic_calibration.get('camera_matrix'),
            distortion_coefficients=intrinsic_calibration.get('distortion_coefficients'),
            remove_behind_camera=True,
            remove_outside_frame=True,
            image_corners=image_corners
        )
        if mark_device_locations:
            device_image_points = cv_utils.core.project_points(
                object_points=device_object_points,
                rotation_vector=extrinsic_calibration.get('rotation_vector'),
                translation_vector=extrinsic_calibration.get('translation_vector'),
                camera_matrix=intrinsic_calibration.get('camera_matrix'),
                distortion_coefficients=intrinsic_calibration.get('distortion_coefficients'),
                remove_behind_camera=True,
                remove_outside_frame=True,
                image_corners=image_corners
            )
        logger.info('Drawing visualization')
        image = cv2.imread(metadatum.get('image_local_path'))
        image = draw_floor_grid_image_points(
            original_image=image,
            image_points=floor_grid_image_points,
            radius=grid_point_radius,
            color=grid_point_color,
            alpha=grid_point_alpha
        )
        image = draw_floor_grid_corner_labels(
            original_image=image,
            image_points=grid_corner_image_points,
            object_points=grid_corner_object_points,
            horizontal_alignment=corner_label_horizontal_alignment,
            vertical_alignment=corner_label_vertical_alignment,
            font_scale=corner_label_font_scale,
            line_width=corner_label_line_width,
            color=corner_label_color,
            alpha=corner_label_alpha
        )
        if mark_device_locations:
            image = draw_device_image_points(
                original_image=image,
                image_points=device_image_points,
                radius=device_point_radius,
                color=device_point_color,
                alpha=device_point_alpha
            )
            image = draw_device_labels(
                original_image=image,
                image_points=device_image_points,
                labels=device_names,
                horizontal_alignment=device_label_horizontal_alignment,
                vertical_alignment=device_label_vertical_alignment,
                font_scale=device_label_font_scale,
                line_width=device_label_line_width,
                color=device_label_color,
                alpha=device_label_alpha
            )
        if show:
            logger.info('Showing visualization')
            cv2.imshow(camera_device_id, image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        if save:
            logger.info('Saving visualization')
            output_filename = 'calibration_{}_{}.{}'.format(
                metadatum.get('device_id'),
                visualization_datetime.astimezone(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'),
                output_filename_extension
            )
            output_path = os.path.join(output_directory, output_filename)
            os.makedirs(output_directory, exist_ok=True)
            cv2.imwrite(output_path, image)


def draw_floor_grid_image_points(
    original_image,
    image_points,
    radius=3,
    line_width=1.5,
    color='#00ff00',
    fill=True,
    alpha=1.0
):
    image_points = np.asarray(image_points).reshape((-1, 2))
    output_image = original_image.copy()
    for point_index in range(image_points.shape[0]):
        point = image_points[point_index]
        if np.any(np.isnan(point)):
            continue
        output_image = cv_utils.drawing.opencv.draw_circle(
            original_image=output_image,
            coordinates=point,
            radius=radius,
            line_width=line_width,
            color=color,
            fill=True,
            alpha=1.0
        )
    return output_image

def draw_device_image_points(
    original_image,
    image_points,
    radius=3,
    line_width=1.5,
    color='#ff0000',
    fill=True,
    alpha=1.0
):
    image_points = np.asarray(image_points).reshape((-1, 2))
    output_image = original_image.copy()
    for point_index in range(image_points.shape[0]):
        point = image_points[point_index]
        if np.any(np.isnan(point)):
            continue
        output_image = cv_utils.drawing.opencv.draw_circle(
            original_image=output_image,
            coordinates=point,
            radius=radius,
            line_width=line_width,
            color=color,
            fill=True,
            alpha=1.0
        )
    return output_image

def draw_floor_grid_corner_labels(
    original_image,
    image_points,
    object_points,
    horizontal_alignment='left',
    vertical_alignment='bottom',
    font_scale=1.0,
    line_width=2,
    color='#00ff00',
    alpha=1.0
):
    output_image = original_image.copy()
    for point_index in range(object_points.shape[0]):
        object_point = object_points[point_index]
        image_point = image_points[point_index]
        if np.any(np.isnan(image_point)):
            continue
        text = '({}, {})'.format(
            round(object_point[0]),
            round(object_point[1])
        )
        output_image = cv_utils.drawing.opencv.draw_text(
            original_image=output_image,
            coordinates=image_point,
            text=text,
            horizontal_alignment=horizontal_alignment,
            vertical_alignment=vertical_alignment,
            font_scale=font_scale,
            line_width=line_width,
            color=color,
            alpha=alpha
        )
    return output_image

def draw_device_labels(
    original_image,
    image_points,
    labels,
    horizontal_alignment='left',
    vertical_alignment='bottom',
    font_scale=1.0,
    line_width=1,
    color='#ff0000',
    alpha=1.0
):
    output_image = original_image.copy()
    for point_index in range(image_points.shape[0]):
        image_point = image_points[point_index]
        if np.any(np.isnan(image_point)):
            continue
        text = labels[point_index]
        output_image = cv_utils.drawing.opencv.draw_text(
            original_image=output_image,
            coordinates=image_point,
            text=text,
            horizontal_alignment=horizontal_alignment,
            vertical_alignment=vertical_alignment,
            font_scale=font_scale,
            line_width=line_width,
            color=color,
            alpha=alpha
        )
    return output_image

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
