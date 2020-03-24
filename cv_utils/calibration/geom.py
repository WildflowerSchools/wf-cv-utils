from collections import OrderedDict
from enum import Enum
import json
import math
import typing

import attr
import cv2 as cv
import numpy as np


@attr.s
class Point2D:
    x: float = attr.ib()
    y: float = attr.ib()
    id3d: int = attr.ib()

    def to_tuple(self):
        return (self.x, self.y)


@attr.s
class Point3D:
    x: float = attr.ib()
    y: float = attr.ib()
    z: float = attr.ib()

    def to_tuple(self):
        return (self.x, self.y, self.z)


@attr.s
class PointMapping:
    image: Point2D = attr.ib()
    world: Point3D = attr.ib()

    def to_dict(self):
        d = OrderedDict((
            ('imageX', self.image.x),
            ('imageY', self.image.y),
            ('worldX', self.world.x),
            ('worldY', self.world.y),
            ('worldZ', self.world.z),
        ))
        return d


class AxisLabel(Enum):
    X = 'x'
    Y = 'y'
    # Z = 'z'


@attr.s
class Axis:
    label: AxisLabel = attr.ib()
    value: float = attr.ib()


@attr.s
class Line:
    pt1: Point2D = attr.ib()
    pt2: Point2D = attr.ib()
    axis: Axis = attr.ib()

    # def plot(self, ax, color: str='red', alpha: float=0.55, linewidth: int=4, **_kwargs):
    #     kwargs = {
    #         'color': color,
    #         'alpha': alpha,
    #         'linewidth': linewidth,
    #     }
    #     kwargs.update(_kwargs)
    #     ax.plot([self.pt1.x, self.pt2.x], [self.pt1.y, self.pt2.y], **kwargs)

    def to_tuple(self):
        return (self.pt1.to_tuple(), self.pt2.to_tuple())

    def intersect(self, other: 'Line'):
        (x1, y1), (x2, y2) = self.to_tuple()
        (x3, y3), (x4, y4) = other.to_tuple()

        t_n = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        t_d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = t_n / t_d
        u_n = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
        u_d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        u = -(u_n / u_d)
        if 0 <= t <= 1:
            Px = x1 + t * (x2 - x1)
            Py = y1 + t * (y2 - y1)
        elif 0 <= u <= 1:
            Px = x3 + u * (x4 - x3)
            Py = y3 + u * (y4 - y3)
        else:
            return None
        image_pt = Point2D(Px, Py)

        if self.axis and self.axis.label == AxisLabel.X:
            x = self.axis.value
        elif other.axis and other.axis.label == AxisLabel.X:
            x = other.axis.value
        else:
            return None

        if self.axis and self.axis.label == AxisLabel.Y:
            y = self.axis.value
        elif other.axis and other.axis.label == AxisLabel.Y:
            y = other.axis.value
        else:
            return None

        x = float(x)
        y = float(y)
        world_pt = Point3D(x, y, 0)
        mapping = PointMapping(image_pt, world_pt)
        return mapping
