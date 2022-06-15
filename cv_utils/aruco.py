import cv2 as cv

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
        self.aruco_dict = fetch_aruco_dictionary(
            num_squares_x=self.num_squares_x,
            num_squares_y=self.num_squares_y,
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

def fetch_aruco_dictionary(
    num_squares_x=7,
    num_squares_y=5,
    marker_size=6,
    april_tag=False
):
    compatible_aruco_dictionaries = list(filter(
        lambda item: (
            item[1]['marker_size']==marker_size and
            item[1]['num_markers'] >= num_squares_x*num_squares_y and
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
