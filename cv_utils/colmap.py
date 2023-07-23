import cv_utils.core
import pandas as pd
import numpy as np
import re
import os
import logging

logger = logging.getLogger(__name__)

CALIBRATION_DATA_RE = r'(?P<colmap_image_id>[0-9]+) (?P<qw>[-0-9.]+) (?P<qx>[-0-9.]+) (?P<qy>[-0-9.]+) (?P<qz>[-0-9.]+) (?P<tx>[-0-9.]+) (?P<ty>[-0-9.]+) (?P<tz>[-0-9.]+) (?P<colmap_camera_id>[0-9]+) (?P<image_path>.+)'

def fetch_colmap_output_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    image_data_path=None,
    camera_data_path=None,
    ref_images_data_path=None
):
    """
    Fetches data from COLMAP input and output files and assembles into dataframe.

    The script essentially executes fetch_colmap_image_data_local(),
    fetch_colmap_camera_data_local(), and
    fetch_colmap_reference_image_data_local(); joins their outputs; calculates
    the difference between the camera position inputs and the camera position
    outputs; and assembles everything into a dataframe.

    For details, see documentation for the constituent functions.

    Args:
        calibration_directory (str): Path to directory containing calibrations
        calibration_identifier (str): Identifier for this particular calibration
        image_data_path (str): Explicit path for COLMAP images output file (default is None)
        camera_data_path (str): Explicit path for COLMAP cameras output file (default is None)
        ref_images_data_path (str): Explicit path for COLMAP ref images input file (default is None)

    Returns:
        (DataFrame) Dataframe containing COLMAP output data
    """
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

def fetch_colmap_camera_data_local(
    calibration_directory=None,
    calibration_identifier=None,
    path=None
):
    """
    Fetches data from COLMAP cameras output file and assembles into dataframe.

    The script parses the COLMAP cameras output file, extracting the COLMAP
    camera ID, COLMAP camera model (e.g., OPENCV), image width, image height,
    and intrinsic calibration parameters for each camera.

    For each camera, it then extracts the camera matrix and distortion
    coefficients from the intrinsic calibration parameters.

    By default, the script assumes that the COLMAP cameras output is in a file
    called cameras.txt in the directory
    calibration_directory/calibration_identifier. These are the also the default
    path and naming conventions for COLMAP. Alternatively, the user can
    explicitly specify the path for the COLMAP cameras output file.

    Args:
        calibration_directory (str): Path to directory containing calibrations
        calibration_identifier (str): Identifier for this particular calibration
        path (str): Explicit path for COLMAP cameras output file (default is
        None)

    Returns:
        (DataFrame) Dataframe containing camera data
    """
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
    df = (
        df
        .reindex(columns=[
            'colmap_camera_id',
            'colmap_camera_model',
            'image_width',
            'image_height',
            'colmap_parameters',
            'camera_matrix',
            'distortion_coefficients'
        ])
        .set_index('colmap_camera_id')
    )
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
    """
    Fetches data from COLMAP ref images input file and assembles into dataframe.

    The script parses the COLMAP ref images input file, extracting the image
    path and (input) camera position for each image (for comparison with the
    calculated camera position).

    By default, the script assumes that the COLMAP ref images input data is in a
    file called ref_images.txt in the directory
    calibration_directory/calibration_identifier. These are the also the default
    path and naming conventions for COLMAP. Alternatively, the user can
    explicitly specify the path for the COLMAP ref images output file.

    Args:
        calibration_directory (str): Path to directory containing calibrations
        calibration_identifier (str): Identifier for this particular calibration
        path (str): Explicit path for COLMAP ref images input file (default is
        None)

    Returns:
        (DataFrame) Dataframe containing camera position input data
    """
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
    df = (
        df
        .reindex(columns=[
            'image_path',
            'position_input',
        ])
        .set_index('image_path')
    )
    return df

