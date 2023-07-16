import logging

logger = logging.getLogger(__name__)

def hex_to_rgb(color_hex):
    red, green, blue = bytes.fromhex(color_hex.lstrip('#'))
    color_rgb = (red, green, blue)
    return color_rgb


def rgb_to_bgr(color_rgb):
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    return color_bgr


def hex_to_bgr(color_hex):
    return rgb_to_bgr(hex_to_rgb(color_hex))
