import cv2 as cv
import cv_utils.color
import math

MARKER_DICT = {
    '+': cv.MARKER_CROSS,
    'x': cv.MARKER_TILTED_CROSS,
    '*': cv.MARKER_STAR,
    'd': cv.MARKER_DIAMOND,
    's': cv.MARKER_SQUARE,
    '^': cv.MARKER_TRIANGLE_UP,
    'v': cv.MARKER_TRIANGLE_DOWN
}

def draw_circle(
    original_image,
    coordinates,
    radius=6,
    line_width=1.5,
    color='#00ff00',
    fill=True,
    alpha=1.0
):
    color_bgr = cv_utils.color.hex_to_bgr(color)
    thickness=math.ceil(line_width)
    if fill:
        thickness = cv.FILLED
    overlay_image = original_image.copy()
    overlay_image = cv.circle(
        img=overlay_image,
        center=tuple(coordinates),
        radius=math.ceil(radius),
        color=color_bgr,
        thickness=thickness,
        lineType=cv.LINE_AA
    )
    new_image = cv.addWeighted(
        overlay_image,
        alpha,
        original_image,
        1 - alpha,
        0
    )
    return new_image

def draw_line(
    original_image,
    coordinates,
    line_width=1.5,
    color='#00ff00',
    alpha=1.0
):
    color_bgr = cv_utils.color.hex_to_bgr(color)
    overlay_image = original_image.copy()
    overlay_image = cv.line(
        img=overlay_image,
        pt1 = tuple(coordinates[0]),
        pt2 = tuple(coordinates[1]),
        color=color_bgr,
        thickness=math.ceil(line_width),
        lineType=cv.LINE_AA
    )
    new_image = cv.addWeighted(
        overlay_image,
        alpha,
        original_image,
        1 - alpha,
        0
    )
    return new_image

def draw_text(
    original_image,
    coordinates,
    text,
    horizontal_alignment='center',
    vertical_alignment='bottom',
    font_face=cv.FONT_HERSHEY_PLAIN,
    font_scale=1.0,
    line_width=1,
    color='#00ff00',
    alpha=1.0
):
    color_bgr = cv_utils.color.hex_to_bgr(color)
    thickness=math.ceil(line_width)
    text_box_size, baseline = cv.getTextSize(
        text=text,
        fontFace=font_face,
        fontScale=font_scale,
        thickness=thickness
    )
    text_box_width, text_box_height = text_box_size
    if horizontal_alignment == 'left':
        org_u = coordinates[0]
    elif horizontal_alignment == 'center':
        org_u = coordinates[0] - text_box_width/2
    elif horizontal_alignment == 'right':
        org_u = coordinates[0] - text_box_width
    else:
        raise ValueError('Horizontal aligment \'{}\' not recognized'.format(horizontal_alignment))
    if vertical_alignment == 'top':
        org_v = coordinates[1] + text_box_height
    elif vertical_alignment == 'middle':
        org_v = coordinates[1] + text_box_height/2
    elif vertical_alignment == 'bottom':
        org_v = coordinates[1]
    else:
        raise ValueError('Vertical aligment \'{}\' not recognized'.format(vertical_alignment))
    overlay_image = original_image.copy()
    overlay_image = cv.putText(
        img=overlay_image,
        text=text,
        org=(round(org_u), round(org_v)),
        fontFace=font_face,
        fontScale=font_scale,
        color=color_bgr,
        thickness=thickness,
        lineType=cv.LINE_AA,
    )
    new_image = cv.addWeighted(
        overlay_image,
        alpha,
        original_image,
        1 - alpha,
        0
    )
    return new_image

def draw_point(
    original_image,
    coordinates,
    marker='.',
    marker_size=10,
    line_width=1,
    color='#00ff00',
    alpha=1.0
):
    if marker == '.':
        return draw_circle(
            original_image,
            coordinates,
            radius=marker_size/2,
            color=color,
            fill=True,
            alpha=alpha
        )
    color_bgr = cv_utils.color.hex_to_bgr(color)
    markerType = MARKER_DICT.get(marker)
    overlay_image = original_image.copy()
    overlay_image = cv.drawMarker(
        img=overlay_image,
        position=tuple(coordinates),
        color=color_bgr,
        markerType=markerType,
        markerSize=math.ceil(marker_size),
        thickness=math.ceil(line_width),
        line_type=cv.LINE_AA
    )
    new_image = cv.addWeighted(
        overlay_image,
        alpha,
        original_image,
        1 - alpha,
        0
    )
    return new_image

def draw_rectangle(
    original_image,
    coordinates,
    line_width=1.5,
    color='#00ff00',
    fill=True,
    alpha=1.0
):
    color_bgr = cv_utils.color.hex_to_bgr(color)
    thickness=math.ceil(line_width)
    if fill:
        thickness = cv.FILLED
    overlay_image = original_image.copy()
    overlay_image = cv.rectangle(
        img=overlay_image,
        pt1 = tuple(coordinates[0]),
        pt2 = tuple(coordinates[1]),
        color=color_bgr,
        thickness=thickness,
        lineType=cv.LINE_AA
    )
    new_image = cv.addWeighted(
        overlay_image,
        alpha,
        original_image,
        1 - alpha,
        0
    )
    return new_image
