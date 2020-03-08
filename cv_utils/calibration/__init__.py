import typing

import attr
import cv2 as cv
import numpy as np

from cv_utils.calibration.geom import Point2D, Point3D, PointMapping, AxisLabel, Axis, Line


__all__ = ["Camera", "Point2D", "Point3D", "PointMapping", "AxisLabel", "Axis", "Line"]


@attr.s
class Camera:
    matrix: typing.List[typing.List[float]] = attr.ib()
    distortion_coefficients: typing.List[float] = attr.ib()

    # @classmethod
    # def 

    def matrix_numpy(self):
        return np.array(self.matrix)

    def distortion_coefficients_numpy(self):
        return np.array(self.distortion_coefficients)

    def calibrate(self, mappings: typing.List[PointMapping]):
        assert len(mappings) >= 4, "4 or more mappings are required to calibrate"
        object_points = [[p.world.x, p.world.y, p.world.z] for p in mappings]
        image_points = [[p.image.x, p.image.y] for p in mappings]
        result, rvec, tvec = cv.solvePnP(
            np.array(object_points).astype(np.float64),
            np.array(image_points).astype(np.float64),
            self.matrix_numpy(),
            self.distortion_coefficients_numpy(),
            None,
            None,
            False,
            cv.SOLVEPNP_ITERATIVE)
        return (result, {
                'cameraMatrix': self.matrix,
                'distortionCoefficients': self.distortion_coefficients,
                'rotationVector': rvec.tolist(),
                'translationVector': tvec.tolist(),
            }, )
