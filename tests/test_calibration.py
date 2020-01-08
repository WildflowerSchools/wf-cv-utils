import pytest

from cv_utils.calibration import Camera
from cv_utils.calibration.geom import Point2D, Point3D, PointMapping


def test_camera_calibration():
    camera = Camera(matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0])
    mappings = [
            PointMapping(image=Point2D(x=798, y=802), world=Point3D(x=0, y=0, z=0)),
            PointMapping(image=Point2D(x=798, y=998), world=Point3D(x=1, y=1, z=0)),
            PointMapping(image=Point2D(x=399, y=1200), world=Point3D(x=1, y=3, z=0)),
            PointMapping(image=Point2D(x=1401, y=1301), world=Point3D(x=4, y=1, z=0)),
        ]
    calibration = camera.calibrate(mappings)
    print(calibration)
    assert calibration is not None
    assert calibration['rotationVector'] is not None
    assert calibration['translationVector'] is not None
