import cv_utils.core
import pandas as pd
import numpy as np
import re
import os
import logging

logger = logging.getLogger(__name__)

CALIBRATION_DATA_RE = r'(?P<colmap_image_id>[0-9]+) (?P<qw>[-0-9.]+) (?P<qx>[-0-9.]+) (?P<qy>[-0-9.]+) (?P<qz>[-0-9.]+) (?P<tx>[-0-9.]+) (?P<ty>[-0-9.]+) (?P<tz>[-0-9.]+) (?P<colmap_camera_id>[0-9]+) (?P<image_path>.+)'


def fetch_colmap_image_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    path=None,
):
    """
    Fetches data from COLMAP images output file and assembles into dataframe.

    The script parses the COLMAP images output file, extracting the COLMAP image
    ID, COLMAP camera ID, image path, quaternion vector, and translation vector
    for each image.

    For each image, it then calculates a rotation vector from the quaternion
    vector; calculates a camera position from the rotation vector and
    translation vector; and parses the image path into its subdirectory,
    filename stem, and filename extension.

    By default, the script assumes that the COLMAP images output is in a file
    called images.txt in the directory
    calibration_directory/calibration_identifier. These are the also the default
    path and naming conventions for COLMAP. Alternatively, the user can
    explicitly specify the path for the COLMAP images output file.

    Args:
        calibration_directory (str): Path to directory containing calibrations
        calibration_identifier (str): Identifier for this particular calibration
        path (str): Explicit path for COLMAP image output file (default is None)

    Returns:
        (DataFrame) Dataframe containing image data
    """
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
    df = (
        df
        .reindex(columns=[
            'colmap_image_id',
            'colmap_camera_id',
            'image_path',
            'image_directory',
            'image_name',
            'image_extension',
            'quaternion_vector',
            'rotation_vector',
            'translation_vector',
            'position'
        ])
        .set_index('colmap_image_id')
    )
    return df


