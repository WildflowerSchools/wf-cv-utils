import pytest

from cv_utils.color import hex_to_rgb, rgb_to_bgr, hex_to_bgr


def test_hex_to_rgb():
    rgb = hex_to_rgb("#000000")
    assert rgb == (0, 0, 0, )
    rgb = hex_to_rgb("#ffffff")
    assert rgb == (255, 255, 255, )
    rgb = hex_to_rgb("#33ef1f")
    assert rgb == (51, 239, 31, )


def test_rgb_to_bgr():
    bgr = rgb_to_bgr((55, 87, 200,))
    assert bgr == (200, 87, 55,)


def test_hex_to_bgr():
    bgr = hex_to_bgr("#33ef1f")
    assert bgr == (31, 239, 51,)
