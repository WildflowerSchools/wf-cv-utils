import cv_utils.core
import cv2 as cv
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


CV_ARUCO_DICTIONARIES = {
    cv.aruco.DICT_4X4_50: {'marker_size': 4, 'num_markers': 50, 'april_tag': False},
    cv.aruco.DICT_4X4_100: {'marker_size': 4, 'num_markers': 100, 'april_tag': False},
    cv.aruco.DICT_4X4_250: {'marker_size': 4, 'num_markers': 250, 'april_tag': False},
    cv.aruco.DICT_4X4_1000: {'marker_size': 4, 'num_markers': 1000, 'april_tag': False},
    cv.aruco.DICT_5X5_50: {'marker_size': 5, 'num_markers': 50, 'april_tag': False},
    cv.aruco.DICT_5X5_100: {'marker_size': 5, 'num_markers': 100, 'april_tag': False},
    cv.aruco.DICT_5X5_250: {'marker_size': 5, 'num_markers': 250, 'april_tag': False},
    cv.aruco.DICT_5X5_1000: {'marker_size': 5, 'num_markers': 1000, 'april_tag': False},
    cv.aruco.DICT_6X6_50: {'marker_size': 6, 'num_markers': 50, 'april_tag': False},
    cv.aruco.DICT_6X6_100: {'marker_size': 6, 'num_markers': 100, 'april_tag': False},
    cv.aruco.DICT_6X6_250: {'marker_size': 6, 'num_markers': 250, 'april_tag': False},
    cv.aruco.DICT_6X6_1000: {'marker_size': 6, 'num_markers': 1000, 'april_tag': False},
    cv.aruco.DICT_7X7_50: {'marker_size': 7, 'num_markers': 50, 'april_tag': False},
    cv.aruco.DICT_7X7_100: {'marker_size': 7, 'num_markers': 100, 'april_tag': False},
    cv.aruco.DICT_7X7_250: {'marker_size': 7, 'num_markers': 250, 'april_tag': False},
    cv.aruco.DICT_7X7_1000: {'marker_size': 7, 'num_markers': 1000, 'april_tag': False},
    cv.aruco.DICT_ARUCO_ORIGINAL: {'marker_size': 5, 'num_markers': 1024, 'april_tag': False},
    cv.aruco.DICT_APRILTAG_16h5: {'marker_size': 4, 'num_markers': 30, 'april_tag': True},
    cv.aruco.DICT_APRILTAG_25h9: {'marker_size': 5, 'num_markers': 35, 'april_tag': True},
    cv.aruco.DICT_APRILTAG_36h10: {'marker_size': 6, 'num_markers': 2320, 'april_tag': True},
    cv.aruco.DICT_APRILTAG_36h11: {'marker_size': 6, 'num_markers': 587, 'april_tag': True}
}

CV_CORNER_REFINEMENT_METHODS = {
    'none': cv.aruco.CORNER_REFINE_NONE,
    'subpixel': cv.aruco.CORNER_REFINE_SUBPIX,
    'contour': cv.aruco.CORNER_REFINE_CONTOUR,
    'april_tag': cv.aruco.CORNER_REFINE_APRILTAG
}

class CharucoBoard:

    def __init__(
        self,
        num_squares_x,
        num_squares_y,
        marker_size,
        square_side_length,
        marker_side_length_ratio,
        aruco_dictionary
    ):
        self.num_squares_x = num_squares_x
        self.num_squares_y = num_squares_y
        self.marker_size = marker_size
        self.square_side_length = square_side_length
        self.marker_side_length_ratio = marker_side_length_ratio
        self.marker_side_length = marker_side_length_ratio*square_side_length
        self.aruco_dictionary = aruco_dictionary
        self._cv_charuco_board = cv.aruco.CharucoBoard_create(
            squaresX=self.num_squares_x,
            squaresY=self.num_squares_y,
            squareLength=self.square_side_length,
            markerLength=self.marker_side_length,
            dictionary=self.aruco_dictionary._cv_aruco_dictionary
        )

    @classmethod
    def from_predefined_dictionary(
        cls,
        num_squares_x=7,
        num_squares_y=5,
        marker_size=6,
        square_side_length=1.0,
        marker_side_length_ratio=0.8,
        april_tag=False
    ):
        logger.info('Creating {}x{} ChArUco board with {}x{} markers, square side length of {}, and marker side length ratio of {} based on pre-defined ArUco dictionary'.format(
            num_squares_x,
            num_squares_y,
            marker_size,
            marker_size,
            square_side_length,
            marker_side_length_ratio
        ))
        aruco_dictionary = ArucoDictionary.get_predefined(
            num_markers=num_squares_x*num_squares_y,
            marker_size=marker_size,
            april_tag=april_tag
        )
        return cls(
            num_squares_x=num_squares_x,
            num_squares_y=num_squares_y,
            marker_size=marker_size,
            square_side_length=square_side_length,
            marker_side_length_ratio=marker_side_length_ratio,
            aruco_dictionary=aruco_dictionary
        )

    @classmethod
    def from_custom_dictionary(
        cls,
        num_squares_x=7,
        num_squares_y=5,
        marker_size=6,
        square_side_length=1.0,
        marker_side_length_ratio=0.8,
        random_seed=0
    ):
        logger.info('Creating {}x{} ChArUco board with {}x{} markers, square side length of {}, and marker side length ratio of {} based on custom ArUco dictionary'.format(
            num_squares_x,
            num_squares_y,
            marker_size,
            marker_size,
            square_side_length,
            marker_side_length_ratio
        ))
        aruco_dictionary = ArucoDictionary.generate_custom(
            num_markers=num_squares_x*num_squares_y,
            marker_size=marker_size,
            base_dictionary=None,
            random_seed=random_seed
        )
        return cls(
            num_squares_x=num_squares_x,
            num_squares_y=num_squares_y,
            marker_size=marker_size,
            square_side_length=square_side_length,
            marker_side_length_ratio=marker_side_length_ratio,
            aruco_dictionary=aruco_dictionary
        )

    def write_image(
        self,
        path,
        image_width=None,
        image_height=None,
        margin_size=0,
        num_border_squares=1
    ):
        image = self.create_image(
            image_width=image_width,
            image_height=image_height,
            margin_size=margin_size,
            num_border_squares=num_border_squares
        )
        logger.info('Writing image to \'{}\''.format(
            path
        ))
        cv_utils.core.write_image(
            image=image,
            path=path
        )

    def create_image(
        self,
        image_width=None,
        image_height=None,
        margin_size=0,
        num_border_squares=1
    ):
        image_aspect_ratio = self.num_squares_x/self.num_squares_y
        if image_width is not None and image_height is not None:
            pass
        elif image_width is not None and image_height is None:
            logger.info('Image width specified as {}. Inferring image height from ChArUco board aspect ratio'.format(
                image_width
            ))
            image_height = round(image_width/image_aspect_ratio)
        elif image_width is None and image_height is not None:
            logger.info('Image height specified as {}. Inferring image width from ChArUco board aspect ratio'.format(
                image_height
            ))
            image_width = round(image_height*image_aspect_ratio)
        else:
            raise ValueError('Must specify either image width or image height (or both)')
        image_size = (image_width, image_height)
        logger.info('Creating {}x{} ChArUco board image with {} border squares and margin of {} pixels'.format(
            image_width,
            image_height,
            num_border_squares,
            margin_size
        ))
        image = self._cv_charuco_board.draw(
            outSize=image_size,
            marginSize=margin_size,
            borderBits= num_border_squares
        )
        return image

    def calibrate_camera(
        self,
        image_directory,
        min_chessboard_corners=3,
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
        camera_matrix_guess=None,
        distortion_coefficients_guess=None,
        calibration_max_iterations=30,
        calibration_accuracy=1e-9,
        min_markers=2,
        marker_corner_refinement_method='none',
        marker_corner_refinement_window_size=5,
        marker_corner_refinement_max_iterations=30,
        marker_corner_refinement_accuracy=0.1,
        marker_detector_parameters=None
    ):
        calibration_termination_criteria = cv_utils.core.termination_criteria(
            max_iterations=calibration_max_iterations,
            accuracy=calibration_accuracy
        )
        calibration_flags, calibration_flag_descriptions = cv_utils.core.camera_calibration_flags(
            use_intrinsic_guess=use_intrinsic_guess,
            fix_principal_point=fix_principal_point,
            fix_aspect_ratio=fix_aspect_ratio,
            zero_tangent_distortion=zero_tangent_distortion,
            fix_focal_length=fix_focal_length,
            fix_k1=fix_k1,
            fix_k2=fix_k2,
            fix_k3=fix_k3,
            fix_k4=fix_k4,
            fix_k5=fix_k5,
            fix_k6=fix_k6,
            rational_model=rational_model,
            thin_prism_model=thin_prism_model,
            fix_s1_s2_s3_s4=fix_s1_s2_s3_s4,
            tilted_model=tilted_model,
            fix_taux_tauy=fix_taux_tauy
        )
        if len(calibration_flag_descriptions) > 0:
            logger.info('Calibrating camera with max iterations {}, accuracy {}, and the following flags: {}'.format(
                calibration_max_iterations,
                calibration_accuracy,
                calibration_flag_descriptions
            ))
        else:
            logger.info('Calibrating camera with max iterations {} and accuracy {}. No calibration flags specified.'.format(
                calibration_max_iterations,
                calibration_accuracy
            ))
        chessboard_corners_list = list()
        chessboard_corner_ids_list = list()
        image_filenames = list()
        image_shape = None
        logger.info('Fetching calibration images from directory \'{}\''.format(
            image_directory
        ))
        with os.scandir(image_directory) as it:
            for directory_entry in it:
                if directory_entry.is_file():
                    logger.info('Fetching calibration image \'{}\''.format(
                        directory_entry.name
                    ))
                    image=cv_utils.read_image(directory_entry.path)
                    if image_shape is None:
                        image_shape = image.shape
                    else:
                        if image.shape != image_shape:
                            raise ValueError('Image directory contains images with different shapes')
                    logger.info('Finding ChArUco board corners in image {}'.format(
                        directory_entry.name
                    ))
                    chessboard_corners, chessboard_corner_ids = self.find_chessboard_corners(
                        image=image,
                        min_markers=min_markers,
                        camera_matrix=None,
                        distortion_coefficients=None,
                        marker_corner_refinement_method=marker_corner_refinement_method,
                        marker_corner_refinement_window_size=marker_corner_refinement_window_size,
                        marker_corner_refinement_max_iterations=marker_corner_refinement_max_iterations,
                        marker_corner_refinement_accuracy=marker_corner_refinement_accuracy,
                        marker_detector_parameters=marker_detector_parameters
                    )
                    if chessboard_corners is not None and chessboard_corners.shape[0] > min_chessboard_corners:
                        logger.info('Found more than {} corners in {}. Adding to calibration data.'.format(
                            min_chessboard_corners,
                            directory_entry.name
                        ))
                        image_filenames.append(directory_entry.name)
                        chessboard_corners_list.append(chessboard_corners.reshape((-1, 1, 2)))
                        chessboard_corner_ids_list.append(chessboard_corner_ids.reshape((-1, 1)))
                    else:
                        logger.info('Found fewer than {} corners. Rejecting image'.format(
                            min_chessboard_corners
                        ))
        logger.info('Calibrating based on ChArUco board corners from {} images'.format(
            len(chessboard_corners_list)
        ))
        reprojection_error, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv.aruco.calibrateCameraCharuco(
            charucoCorners=chessboard_corners_list,
            charucoIds=chessboard_corner_ids_list,
            board=self._cv_charuco_board,
            imageSize=(image_shape[1], image_shape[0]),
            cameraMatrix=camera_matrix_guess,
            distCoeffs=distortion_coefficients_guess,
            flags=calibration_flags,
            criteria=calibration_termination_criteria
        )
        extrinsic_parameters = dict()
        for image_filename, rotation_vector, translation_vector in zip(image_filenames, rotation_vectors, translation_vectors):
            extrinsic_parameters[image_filename] = {
                'rotation_vector': rotation_vector,
                'translation_vector': translation_vector
            }
        return camera_matrix, distortion_coefficients, extrinsic_parameters

    def find_chessboard_corners(
        self,
        image,
        min_markers=2,
        camera_matrix=None,
        distortion_coefficients=None,
        marker_corner_refinement_method='none',
        marker_corner_refinement_window_size=5,
        marker_corner_refinement_max_iterations=30,
        marker_corner_refinement_accuracy=0.1,
        marker_detector_parameters=None
    ):
        logger.info('Finding ChArUco board corners in {}x{} image. Minimum of {} detected markers for each corner. Detecting markers.'.format(
            image.shape[1],
            image.shape[0],
            min_markers
        ))
        marker_corners, marker_ids, rejected_image_points = self.detect_markers(
            image=image,
            corner_refinement_method=marker_corner_refinement_method,
            corner_refinement_window_size=marker_corner_refinement_window_size,
            corner_refinement_max_iterations=marker_corner_refinement_max_iterations,
            corner_refinement_accuracy=marker_corner_refinement_accuracy,
            detector_parameters=marker_detector_parameters
        )
        logger.info('Interpolating corners')
        num_chessboard_corners, chessboard_corners, chessboard_corner_ids = cv.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=image,
            board=self._cv_charuco_board,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
            minMarkers=min_markers
        )
        chessboard_corners = np.squeeze(chessboard_corners).reshape((-1, 2))
        chessboard_corner_ids = np.squeeze(chessboard_corner_ids).reshape((-1))
        if chessboard_corners is not None and num_chessboard_corners > 1:
            logger.info('Found {} ChArUco board corners (ids {})'.format(
                num_chessboard_corners,
                ', '.join([str(chessboard_corner_id) for chessboard_corner_id in np.sort(chessboard_corner_ids)])
            ))
        elif chessboard_corners is not None and num_chessboard_corners == 1:
            logger.info('Found 1 ChArUco board corners (id {})'.format(
                chessboard_corner_ids[0]
            ))
        else:
            logger.info('Found no ChArUco board corners')
        return chessboard_corners, chessboard_corner_ids

    def detect_markers(
        self,
        image,
        corner_refinement_method='none',
        corner_refinement_window_size=5,
        corner_refinement_max_iterations=30,
        corner_refinement_accuracy=0.1,
        detector_parameters=None
    ):
        return self.aruco_dictionary.detect_markers(
            image=image,
            corner_refinement_method=corner_refinement_method,
            corner_refinement_window_size=corner_refinement_window_size,
            corner_refinement_max_iterations=corner_refinement_max_iterations,
            corner_refinement_accuracy=corner_refinement_accuracy,
            detector_parameters=detector_parameters
        )


class ArucoDictionary:
    def __init__(
        self,
        num_markers,
        marker_size,
        _cv_aruco_dictionary
    ):
        self.num_markers = num_markers
        self.marker_size = marker_size
        self._cv_aruco_dictionary = _cv_aruco_dictionary

    @classmethod
    def get_predefined(
        cls,
        num_markers=40,
        marker_size=6,
        april_tag=False
    ):
        logger.info('Looking for pre-defined ArUco dictionaries compatible with {} {}x{} markers (AprilTag = {})'.format(
            num_markers,
            marker_size,
            marker_size,
            april_tag
        ))
        compatible_cv_aruco_dictionaries = list(filter(
            lambda item: (
                item[1]['marker_size']==marker_size and
                item[1]['num_markers'] >= num_markers and
                item[1]['april_tag'] == april_tag
            ),
            CV_ARUCO_DICTIONARIES.items()
        ))
        if len(compatible_cv_aruco_dictionaries) == 0:
            raise ValueError('No predefined Aruco dictionaries are compatible with the specified parameters')
        elif len(compatible_cv_aruco_dictionaries) == 1:
            selected_cv_aruco_dictionary = compatible_cv_aruco_dictionaries[0]
            logger.info('Only one compatible pre-defined ArUco dictionary found')
        else:
            selected_cv_aruco_dictionary = min(
                compatible_cv_aruco_dictionaries,
                key=lambda x: x[1]['num_markers']
            )
            logger.info('{} compatible pre-defined ArUco dictionaries found. Selected smallest compatible dictionary ({} markers)'.format(
                len(compatible_cv_aruco_dictionaries),
                selected_cv_aruco_dictionary[1]['num_markers']
            ))
        cv_aruco_dictionary_specifier = selected_cv_aruco_dictionary[0]
        cv_aruco_dictionary = cv.aruco.Dictionary_get(cv_aruco_dictionary_specifier)
        return cls(
            num_markers=num_markers,
            marker_size=marker_size,
            _cv_aruco_dictionary=cv_aruco_dictionary
        )

    @classmethod
    def generate_custom(
        cls,
        num_markers=40,
        marker_size=6,
        base_dictionary=None,
        random_seed=0
    ):
        if base_dictionary is not None:
            logger.info('Generating custom ArUco dictionary with {} {}x{} markers based on dictionary with {} {}x{} markers'.format(
                num_markers,
                marker_size,
                marker_size,
                base_dictionary.num_markers,
                base_dictionary.marker_size,
                base_dictionary.marker_size
            ))
            cv_aruco_dictionary = cv.aruco.custom_dictionary_from(
                nMarkers=num_markers,
                markerSize=marker_size,
                baseDictionary=base_dictionary._cv_aruco_dictionary,
                randomSeed=random_seed
            )
        else:
            logger.info('Generating custom ArUco dictionary with {} {}x{} markers'.format(
                num_markers,
                marker_size,
                marker_size
            ))
            cv_aruco_dictionary = cv.aruco.custom_dictionary(
                nMarkers=num_markers,
                markerSize=marker_size,
                randomSeed=random_seed
            )
        return cls(
            num_markers=num_markers,
            marker_size=marker_size,
            _cv_aruco_dictionary=cv_aruco_dictionary
        )

    def bytes_list(self):
        return self._cv_aruco_dictionary.bytesList

    def max_correction_bits(self):
        return self._cv_aruco_dictionary.maxCorrectionBits

    def write_image(
        self,
        path,
        id,
        image_size=1000,
        num_border_squares=1
    ):
        image = self.create_image(
            id=id,
            image_size=image_size,
            num_border_squares=num_border_squares
        )
        logger.info('Writing image to \'{}\''.format(
            path
        ))
        cv_utils.core.write_image(
            image=image,
            path=path
        )

    def create_image(
        self,
        id,
        image_size=1000,
        num_border_squares=1
    ):
        logger.info('Creating {}x{} image of marker {} with {} border squares'.format(
            image_size,
            image_size,
            id,
            num_border_squares
        ))
        image=self._cv_aruco_dictionary.drawMarker(
            id=id,
            sidePixels=image_size,
            borderBits=num_border_squares
        )
        return image

    def detect_markers(
        self,
        image,
        corner_refinement_method='none',
        corner_refinement_window_size=5,
        corner_refinement_max_iterations=30,
        corner_refinement_accuracy=0.1,
        detector_parameters=None
    ):
        corner_refinement_method_specifier = CV_CORNER_REFINEMENT_METHODS.get(corner_refinement_method)
        if corner_refinement_method_specifier is None:
            raise ValueError('Corner refinement method must be one of the following: {}'.format(
                ', '.join(CV_CORNER_REFINEMENT_METHODS.keys())
            ))
        detector_parameters_object = cv.aruco.DetectorParameters.create()
        if detector_parameters is not None:
            for parameter, value in detector_parameters.items():
                setattr(detector_parameters_object, parameter, value)
        setattr(detector_parameters_object, 'cornerRefinementMethod', corner_refinement_method_specifier)
        setattr(detector_parameters_object, 'cornerRefinementWinSize', corner_refinement_window_size)
        setattr(detector_parameters_object, 'cornerRefinementMaxIterations', corner_refinement_max_iterations)
        setattr(detector_parameters_object, 'cornerRefinementMinAccuracy', corner_refinement_accuracy)
        image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        logger.info('Detecting ArUco markers with corner refinement method \'{}\', window size {}, max iterations {}, and accuracy {}'.format(
            corner_refinement_method,
            corner_refinement_window_size,
            corner_refinement_max_iterations,
            corner_refinement_accuracy
        ))
        marker_corners, marker_ids, rejected_image_points = cv.aruco.detectMarkers(
            image=image_grayscale,
            dictionary=self._cv_aruco_dictionary,
            parameters=detector_parameters_object
        )
        marker_corners = np.squeeze(np.stack(marker_corners)).reshape((-1, 4, 2))
        marker_ids = np.squeeze(marker_ids).reshape((-1))
        rejected_image_points = np.squeeze(np.stack(rejected_image_points)).reshape((-1, 4, 2))
        if marker_corners is not None and marker_corners.shape[0] > 1:
            logger.info('Detected {} ArUco markers (ids {}). Rejected {} image points.'.format(
                marker_corners.shape[0],
                ', '.join([str(marker_id) for marker_id in np.sort(marker_ids)]),
                rejected_image_points.shape[0]
            ))
        elif marker_corners is not None and marker_corners.shape[0] == 1:
            logger.info('Detected 1 ArUco marker (id {}). Rejected {} image points.'.format(
                marker_ids[0],
                rejected_image_points.shape[0]
            ))
        else:
            logger.info('Detected no ArUco markers')
        return marker_corners, marker_ids, rejected_image_points
