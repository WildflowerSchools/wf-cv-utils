import cv_utils.core
import pandas as pd
import numpy as np
import os

def fetch_colmap_image_data_local(path):
    num_header_lines = 4
    with open(path) as fp:
        num_lines = len(fp.readlines())
    num_data_lines = num_lines - num_header_lines
    if num_data_lines % 2 != 0:
        raise ValueError('File does not have even number of data lines')
    num_images = num_data_lines // 2
    skiprows = (
        list(range(num_header_lines)) +
        [num_header_lines + image_index*2 + 1 for image_index in range(num_images)]
    )
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        header=None,
        skiprows=skiprows,
        names = ['colmap_image_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'colmap_camera_id', 'image_path'],
        dtype={
            'colmap_image_id': 'int',
            'qw': 'float',
            'qx': 'float',
            'qy': 'float',
            'qz': 'float',
            'tx': 'float',
            'ty': 'float',
            'tz': 'float',
            'colmap_camera_id': 'int',
            'image_path': 'string'
        }
    )
    df['quaternion_vector'] = df.apply(
        lambda row: np.asarray([row['qw'], row['qx'], row['qy'], row['qz']]),
        axis=1
    )
    df['translation_vector'] = df.apply(
        lambda row: np.asarray([row['tx'], row['ty'], row['tz']]),
        axis=1
    )
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
    df.set_index('colmap_image_id', inplace=True)
    df = df.reindex(columns=[
        'image_path',
        'image_directory',
        'image_name',
        'image_extension',
        'colmap_camera_id',
        'quaternion_vector',
        'rotation_vector',
        'translation_vector',
        'position'
    ])
    return df

def fetch_colmap_camera_data_local(path):
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

def fetch_colmap_reference_image_data_local(path):
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
