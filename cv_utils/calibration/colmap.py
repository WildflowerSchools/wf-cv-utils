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
