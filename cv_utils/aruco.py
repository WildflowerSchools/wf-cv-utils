import cv_utils.core
import cv2 as cv
import numpy as np

ARUCO_DICTIONARIES = {
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

CORNER_REFINEMENT_METHODS = {
    'none': cv.aruco.CORNER_REFINE_NONE,
    'subpixel': cv.aruco.CORNER_REFINE_SUBPIX,
    'contour': cv.aruco.CORNER_REFINE_CONTOUR,
    'april_tag': cv.aruco.CORNER_REFINE_APRILTAG
}

class CharucoBoard:

    def __init__(
        self,
        num_squares_x=7,
        num_squares_y=5,
        marker_size=6,
        april_tag=False,
        square_side_length=1.0,
        marker_side_length_ratio=0.8
    ):
        self.num_squares_x = num_squares_x
        self.num_squares_y = num_squares_y
        self.marker_size = marker_size
        self.april_tag = april_tag
        self.square_side_length = square_side_length
        self.marker_side_length_ratio = marker_side_length_ratio
        self.marker_side_length = marker_side_length_ratio*square_side_length
        self.aruco_dict = get_predefined_aruco_dictionary(
            num_markers=self.num_squares_x*self.num_squares_y,
            marker_size=self.marker_size,
            april_tag=self.april_tag
        )
        self._cv_charuco_board = cv.aruco.CharucoBoard_create(
            squaresX=self.num_squares_x,
            squaresY=self.num_squares_y,
            squareLength=self.square_side_length,
            markerLength=self.marker_side_length,
            dictionary=self.aruco_dict
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
            image_size = (image_width, image_height)
        elif image_width is not None and image_height is None:
            image_size = (image_width, round(image_width/image_aspect_ratio))
        elif image_width is None and image_height is not None:
            image_size = (round(image_height*image_aspect_ratio), image_height)
        else:
            raise ValueError('Must specify either image width or image height (or both)')
        image = self._cv_charuco_board.draw(
            outSize=image_size,
            marginSize=margin_size,
            borderBits= num_border_squares
        )
        return image

def get_predefined_aruco_dictionary(
    num_markers=40,
    marker_size=6,
    april_tag=False
):
    compatible_aruco_dictionaries = list(filter(
        lambda item: (
            item[1]['marker_size']==marker_size and
            item[1]['num_markers'] >= num_markers and
            item[1]['april_tag'] == april_tag
        ),
        ARUCO_DICTIONARIES.items()
    ))
    if len(compatible_aruco_dictionaries) == 0:
        raise ValueError('No predefined Aruco dictionaries are compatible with the specified parameters')
    elif len(compatible_aruco_dictionaries) == 1:
        selected_aruco_dictionary = compatible_aruco_dictionaries[0]
    else:
        selected_aruco_dictionary = min(
            compatible_aruco_dictionaries,
            key=lambda x: x[1]['num_markers']
        )
    aruco_dict_specifier = selected_aruco_dictionary[0]
    aruco_dictionary = cv.aruco.Dictionary_get(aruco_dict_specifier)
    return aruco_dictionary

def detect_markers(
    image,
    aruco_dict,
    corner_refinement_method='none',
    corner_refinement_window_size=5,
    corner_refinement_max_iterations=30,
    corner_refinement_accuracy=0.1,
    detector_parameters=None
):
    corner_refinement_method_specifier = CORNER_REFINEMENT_METHODS.get(corner_refinement_method)
    if corner_refinement_method_specifier is None:
        raise ValueError('Corner refinement method must be one of the following: {}'.format(
            ', '.join(CORNER_REFINEMENT_METHODS.keys())
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
    corners, ids, rejected_image_points = cv.aruco.detectMarkers(
        image=image_grayscale,
        dictionary=aruco_dict,
        parameters=detector_parameters_object
    )
    corners = np.squeeze(np.stack(corners))
    ids = np.squeeze(ids)
    rejected_image_points = np.squeeze(np.stack(rejected_image_points))
    return corners, ids, rejected_image_points