import cv_utils.core
import cv_utils.calibration.honeycomb
import video_io
import pandas as pd
import numpy as np
import shutil
import re
import os
import logging

logger = logging.getLogger(__name__)


CALIBRATION_DATA_RE = r'(?P<colmap_image_id>[0-9]+) (?P<qw>[-0-9.]+) (?P<qx>[-0-9.]+) (?P<qy>[-0-9.]+) (?P<qz>[-0-9.]+) (?P<tx>[-0-9.]+) (?P<ty>[-0-9.]+) (?P<tz>[-0-9.]+) (?P<colmap_camera_id>[0-9]+) (?P<image_path>.+)'

def write_colmap_output_honeycomb(
    colmap_output_df,
    calibration_start,
    coordinate_space_id,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    calibration_data_columns = [
        'device_id',
        'image_width',
        'image_height',
        'camera_matrix',
        'distortion_coefficients',
        'rotation_vector',
        'translation_vector',
        'position'
    ]
    if not set(calibration_data_columns).issubset(set(colmap_output_df.columns)):
        raise ValueError('COLMAP output data must contain the following columns: {}'.format(
            calibration_data_columns
        ))
    colmap_output_df = colmap_output_df.dropna(subset=['device_id'])
    calibration_start = pd.to_datetime(calibration_start, utc=True).to_pydatetime()
    intrinsic_calibration_ids = cv_utils.calibration.honeyomb.write_intrinsic_calibration_data(
        data=colmap_output_df,
        start_datetime=calibration_start,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    extrinsic_calibration_ids = cv_utils.calibration.honeycomb.write_extrinsic_calibration_data(
        data=colmap_output_df,
        start_datetime=calibration_start,
        coordinate_space_id=coordinate_space_id,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    position_assignment_ids = cv_utils.calibration.honeycomb.write_position_data(
        data=colmap_output_df,
        start_datetime=calibration_start,
        coordinate_space_id=coordinate_space_id,
        assigned_type='DEVICE',
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    honeycomb_ids = {
        'intrinsic_calibration_ids': intrinsic_calibration_ids,
        'extrinsic_calibraion_ids': extrinsic_calibration_ids,
        'position_assignment_ids': position_assignment_ids
    }
    return honeycomb_ids

def prepare_colmap_inputs(
    calibration_directory=None,
    calibration_identifier=None,
    image_info_path=None,
    images_directory_path=None,
    ref_images_data_path=None,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    local_image_directory='./images',
    image_filename_extension='png',
    local_video_directory='./videos',
    video_filename_extension='mp4'
):
    # Set input and output paths
    if image_info_path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either image info path or calibration directory and calibration identifier')
        image_info_path = os.path.join(
            calibration_directory,
            calibration_identifier,
            'image_info.csv'
        )
    if images_directory_path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either image directory path or calibration directory and calibration identifier')
        images_directory_path = os.path.join(
            calibration_directory,
            calibration_identifier,
            'images'
        )
    if ref_images_data_path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either ref image data path or calibration directory and calibration identifier')
        ref_images_data_directory = os.path.join(
            calibration_directory,
            calibration_identifier
        )
        ref_images_data_filename = 'ref_images.txt'
    else:
        ref_images_data_directory = os.path.dirname(os.path.normpath(ref_images_data_path))
        ref_images_data_filename = os.path.basename(os.path.normpath(ref_images_data_path))
    # Fetch image info from CSV file
    image_info_columns = [
        'device_id',
        'camera_type',
        'image_timestamp'
    ]
    image_info_df = pd.read_csv(image_info_path)
    if not set(image_info_columns).issubset(set(image_info_df.columns)):
        raise ValueError('Image info CSV data must contain the following columns: {}'.format(
            image_info_columns
        ))
    image_info_df['image_timestamp'] = pd.to_datetime(image_info_df['image_timestamp'])
    ref_images_lines = list()
    for index, camera in image_info_df.iterrows():
        camera_device_id = camera['device_id']
        image_timestamp = camera['image_timestamp']
        camera_type = camera['camera_type']
        image_metadata = video_io.fetch_images(
            image_timestamps=[image_timestamp],
            camera_device_ids=[camera_device_id],
            chunk_size=chunk_size,
            minimal_honeycomb_client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret,
            local_image_directory=local_image_directory,
            image_filename_extension=image_filename_extension,
            local_video_directory=local_video_directory,
            video_filename_extension=video_filename_extension
        )
        if len(image_metadata) > 1:
            raise ValueError('More than one image returned for this camera and timestamp')
        image_info = image_metadata[0]
        source_path = image_info['image_local_path']
        # Copy image file
        output_directory = os.path.join(
            images_directory_path,
            camera_type
        )
        output_filename = '{}.{}'.format(
            camera_device_id,
            image_filename_extension
        )
        output_path = os.path.join(
            output_directory,
            output_filename
        )
        os.makedirs(output_directory, exist_ok=True)
        shutil.copy2(source_path, output_path)
        # Fetch camera position
        position = cv_utils.calibration.honeycomb.fetch_device_position(
            device_id=camera_device_id,
            datetime=image_timestamp,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
        if position is not None:
            ref_images_line = ' '.join([
                os.path.join(
                    camera_type,
                    output_filename
                ),
                str(position[0]),
                str(position[1]),
                str(position[2])
            ])
            ref_images_lines.append(ref_images_line)
    os.makedirs(ref_images_data_directory, exist_ok=True)
    ref_images_path = os.path.join(
        ref_images_data_directory,
        ref_images_data_filename
    )
    with open(ref_images_path, 'w') as fp:
        fp.write('\n'.join(ref_images_lines))

def fetch_colmap_output_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    image_data_path=None,
    camera_data_path=None,
    ref_images_data_path=None
):
    # Fetch COLMAP image output
    df = fetch_colmap_image_data_local(
        calibration_directory=calibration_directory,
        calibration_identifier=calibration_identifier,
        path=image_data_path
    )
    # Fetch COLMAP cameras output
    cameras_df = fetch_colmap_camera_data_local(
        calibration_directory=calibration_directory,
        calibration_identifier=calibration_identifier,
        path=camera_data_path
    )
    df = df.join(cameras_df, on='colmap_camera_id')
    # Fetch COLMAP ref images input
    ref_images_df = fetch_colmap_reference_image_data_local(
        calibration_directory=calibration_directory,
        calibration_identifier=calibration_identifier,
        path=ref_images_data_path
    )
    df = df.join(ref_images_df, on='image_path')
    # Calculate fields
    df['image_path'] = df['image_path'].astype('string')
    df['position_error'] = df['position'] - df['position_input']
    df['position_error_distance'] = df['position_error'].apply(np.linalg.norm)
    return df

def fetch_colmap_image_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    path=None,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either image data path or calibration directory and calibration identifier')
        path = os.path.join(
            calibration_directory,
            calibration_identifier,
            'images.txt'
        )
    data_list = list()
    with open(path, 'r') as fp:
        for line in fp.readlines():
            m = re.match(CALIBRATION_DATA_RE, line)
            if m:
                data_list.append({
                    'colmap_image_id': int(m.group('colmap_image_id')),
                    'quaternion_vector': np.asarray([
                        float(m.group('qw')),
                        float(m.group('qx')),
                        float(m.group('qy')),
                        float(m.group('qz'))
                    ]),
                    'translation_vector': np.asarray([
                        float(m.group('tx')),
                        float(m.group('ty')),
                        float(m.group('tz'))
                    ]),
                    'colmap_camera_id': int(m.group('colmap_camera_id')),
                    'image_path': m.group('image_path')

                })
    df = pd.DataFrame(data_list)
    df['rotation_vector'] = df['quaternion_vector'].apply(cv_utils.core.quaternion_vector_to_rotation_vector)
    df['position'] = df.apply(
        lambda row: cv_utils.core.extract_camera_position(
            row['rotation_vector'],
            row['translation_vector']
        ),
        axis=1
    )
    df['image_directory'] = df['image_path'].apply(lambda x: os.path.dirname(os.path.normpath(x))).astype('string')
    df['image_name'] = df['image_path'].apply(lambda x: os.path.splitext(os.path.basename(os.path.normpath(x)))[0]).astype('string')
    df['image_extension'] = df['image_path'].apply(
        lambda x: os.path.splitext(os.path.basename(os.path.normpath(x)))[1][1:]
        if len(os.path.splitext(os.path.basename(os.path.normpath(x)))[1]) > 1
        else None
    ).astype('string')
    logger.info('Attempting to extract camera device IDs from image names')
    df['device_id'] = df['image_name'].apply(cv_utils.calibration.honeycomb.extract_honeycomb_id).astype('object')
    device_ids = df['device_id'].dropna().unique().tolist()
    logger.info('Found {} device IDs among image names'.format(
        len(device_ids)
    ))
    logger.info('Fetching camera names')
    camera_names = cv_utils.calibration.honeycomb.fetch_camera_names(
        camera_ids=device_ids,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    df = df.join(
        pd.Series(camera_names, name='camera_name'),
        on='device_id'
    )
    df.set_index('colmap_image_id', inplace=True)
    df = df.reindex(columns=[
        'image_path',
        'image_directory',
        'image_name',
        'device_id',
        'camera_name',
        'image_extension',
        'colmap_camera_id',
        'quaternion_vector',
        'rotation_vector',
        'translation_vector',
        'position'
    ])
    return df

def fetch_colmap_camera_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    path=None
):
    if path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either camera data path or calibration directory and calibration identifier')
        path = os.path.join(
            calibration_directory,
            calibration_identifier,
            'cameras.txt'
        )
    cameras=list()
    with open(path, 'r') as fp:
        for line_index, line in enumerate(fp):
            if len(line) == 0 or line[0] == '#':
                continue
            word_list = line.split()
            if len(word_list) < 5:
                raise ValueError('Line {} is shorter than expected: {}'.format(
                    line_index,
                    line
                ))
            camera = {
                'colmap_camera_id': int(word_list[0]),
                'colmap_camera_model': word_list[1],
                'image_width': int(word_list[2]),
                'image_height': int(word_list[3]),
                'colmap_parameters': np.asarray([float(parameter_string) for parameter_string in word_list[4:]])
            }
            cameras.append(camera)
    df = pd.DataFrame.from_records(cameras)
    df['camera_matrix'] = df.apply(
        lambda row: colmap_parameters_to_opencv_parameters(
            row['colmap_parameters'],
            row['colmap_camera_model']
        )[0],
        axis=1
    )
    df['distortion_coefficients'] = df.apply(
        lambda row: colmap_parameters_to_opencv_parameters(
            row['colmap_parameters'],
            row['colmap_camera_model']
        )[1],
        axis=1
    )
    df = df.astype({
        'colmap_camera_id': 'int',
        'colmap_camera_model': 'string',
        'image_width': 'int',
        'image_height': 'int',
        'colmap_parameters': 'object',
        'camera_matrix': 'object',
        'distortion_coefficients': 'object'
    })
    df.set_index('colmap_camera_id', inplace=True)
    df = df.reindex(columns=[
        'colmap_camera_model',
        'image_width',
        'image_height',
        'colmap_parameters',
        'camera_matrix',
        'distortion_coefficients'
    ])
    return df

def colmap_parameters_to_opencv_parameters(colmap_parameters, colmap_camera_model):
    if colmap_camera_model == 'SIMPLE_PINHOLE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[0]
        cx = colmap_parameters[1]
        cy = colmap_parameters[2]
        distortion_coefficients = None
    elif colmap_camera_model == 'PINHOLE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[1]
        cx = colmap_parameters[2]
        cy = colmap_parameters[3]
        distortion_coefficients = None
    elif colmap_camera_model == 'SIMPLE_RADIAL':
        fx = colmap_parameters[0]
        fy = colmap_parameters[0]
        cx = colmap_parameters[1]
        cy = colmap_parameters[2]
        distortion_coefficients = np.array([
            colmap_parameters[3],
            0.0,
            0.0,
            0.0
        ])
    elif colmap_camera_model == 'RADIAL':
        fx = colmap_parameters[0]
        fy = colmap_parameters[0]
        cx = colmap_parameters[1]
        cy = colmap_parameters[2]
        distortion_coefficients = np.array([
            colmap_parameters[3],
            colmap_parameters[4],
            0.0,
            0.0
        ])
    elif colmap_camera_model == 'OPENCV':
        fx = colmap_parameters[0]
        fy = colmap_parameters[1]
        cx = colmap_parameters[2]
        cy = colmap_parameters[3]
        distortion_coefficients = np.array([
            colmap_parameters[4],
            colmap_parameters[5],
            colmap_parameters[6],
            colmap_parameters[7]
        ])
    elif colmap_camera_model == 'OPENCV_FISHEYE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[1]
        cx = colmap_parameters[2]
        cy = colmap_parameters[3]
        distortion_coefficients = np.array([
            colmap_parameters[4],
            colmap_parameters[5],
            0.0,
            0.0,
            colmap_parameters[6],
            colmap_parameters[7],
            0.0,
            0.0
        ])
    elif colmap_camera_model == 'FULL_OPENCV':
        fx = colmap_parameters[0]
        fy = colmap_parameters[1]
        cx = colmap_parameters[2]
        cy = colmap_parameters[3]
        distortion_coefficients = np.array([
            colmap_parameters[4],
            colmap_parameters[5],
            colmap_parameters[6],
            colmap_parameters[7],
            colmap_parameters[8],
            colmap_parameters[9],
            colmap_parameters[10],
            colmap_parameters[11]
        ])
    elif colmap_camera_model == 'SIMPLE_RADIAL_FISHEYE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[0]
        cx = colmap_parameters[1]
        cy = colmap_parameters[2]
        distortion_coefficients = np.array([
            colmap_parameters[3],
            0.0,
            0.0,
            0.0
        ])
    elif colmap_camera_model == 'RADIAL_FISHEYE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[0]
        cx = colmap_parameters[1]
        cy = colmap_parameters[2]
        distortion_coefficients = np.array([
            colmap_parameters[3],
            colmap_parameters[4],
            0.0,
            0.0
        ])
    elif colmap_camera_model == 'THIN_PRISM_FISHEYE':
        fx = colmap_parameters[0]
        fy = colmap_parameters[1]
        cx = colmap_parameters[2]
        cy = colmap_parameters[3]
        distortion_coefficients = np.array([
            colmap_parameters[4],
            colmap_parameters[5],
            colmap_parameters[6],
            colmap_parameters[7],
            colmap_parameters[8],
            colmap_parameters[9],
            0.0,
            0.0,
            colmap_parameters[10],
            colmap_parameters[11],
            0.0,
            0.0
        ])
    else:
        raise ValueError('Camera model {} not found'.format(colmap_camera_model))
    camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])
    return camera_matrix, distortion_coefficients

def fetch_colmap_reference_image_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    path=None
):
    if path is None:
        if calibration_directory is None or calibration_identifier is None:
            raise ValueError('Must specify either ref image data path or calibration directory and calibration identifier')
        path = os.path.join(
            calibration_directory,
            calibration_identifier,
            'ref_images.txt'
        )
    df = pd.read_csv(
        path,
        header=None,
        delim_whitespace=True,
        names = ['image_path', 'x', 'y', 'z'],
        dtype={
            'image_path': 'string',
            'x': 'float',
            'y': 'float',
            'z': 'float',
        }
    )
    df['position_input'] = df.apply(
        lambda row: np.array([row['x'], row['y'], row['z']]),
        axis=1
    )
    df.set_index('image_path', inplace=True)
    df = df.reindex(columns=[
        'position_input'
    ])
    return df

def extract_colmap_image_calibration_data(
    input_path,
    output_path
):
    output_lines = list()
    with open(input_path, 'r') as fp:
        for line in fp.readlines():
            m = re.match(CALIBRATION_DATA_RE, line)
            if m:
                output_line = ','.join([
                    m.group('colmap_image_id'),
                    m.group('qw'),
                    m.group('qx'),
                    m.group('qy'),
                    m.group('qz'),
                    m.group('tx'),
                    m.group('ty'),
                    m.group('tz'),
                    m.group('colmap_camera_id'),
                    m.group('image_path')
                ])
                output_lines.append(output_line)
    with open(output_path, 'w') as fp:
        fp.write('\n'.join(output_lines))
