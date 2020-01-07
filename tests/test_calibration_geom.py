import pytest

from cv_utils.calibration.geom import Point2D, Point3D, PointMapping, Line, AxisLabel, Axis


def test_point2d_correctness():
    point = Point2D(x=23, y=10)
    assert isinstance(point, Point2D)
    assert point.x == 23
    assert point.y == 10
    assert point.to_tuple() == (23, 10,)


def test_point3d_correctness():
    point = Point3D(x=23, y=10, z=4)
    assert isinstance(point, Point3D)
    assert point.x == 23
    assert point.y == 10
    assert point.z == 4
    assert point.to_tuple() == (23, 10, 4,)


def test_pointmapping_correctness():
    point2d = Point2D(x=1, y=2)
    point3d = Point3D(x=10, y=10, z=40)
    mapping = PointMapping(image=point2d, world=point3d)
    assert mapping.image.x == point2d.x
    assert mapping.image.y == point2d.y
    assert mapping.world.x == point3d.x
    assert mapping.world.y == point3d.y
    assert mapping.world.z == point3d.z
    mapping_dict = mapping.to_dict()
    assert mapping_dict['imageX'] == point2d.x
    assert mapping_dict['imageY'] == point2d.y
    assert mapping_dict['worldX'] == point3d.x
    assert mapping_dict['worldY'] == point3d.y
    assert mapping_dict['worldZ'] == point3d.z


def test_line_correctness():
    line1 = Line(pt1=Point2D(x=10, y=0), pt2=Point2D(x=10, y=10), axis=Axis(AxisLabel.X, 0))
    line2 = Line(pt1=Point2D(x=0, y=5), pt2=Point2D(x=30, y=5), axis=Axis(AxisLabel.Y, 0))
    assert line1.pt1.x == 10
    assert line1.pt1.y == 0
    assert line1.pt2.x == 10
    assert line1.pt2.y == 10
    assert line2.pt1.x == 0
    assert line2.pt1.y == 5
    assert line2.pt2.x == 30
    assert line2.pt2.y == 5
    assert line1.to_tuple() == ((10, 0,), (10, 10,))
    intersect = line1.intersect(line2)
    mapping_dict = intersect.to_dict()
    assert mapping_dict['imageX'] == 10
    assert mapping_dict['imageY'] == 5
    assert mapping_dict['worldX'] == 0
    assert mapping_dict['worldY'] == 0
    assert mapping_dict['worldZ'] == 0
    # test the inverse
    intersect = line2.intersect(line1)
    mapping_dict = intersect.to_dict()
    assert mapping_dict['imageX'] == 10
    assert mapping_dict['imageY'] == 5
    assert mapping_dict['worldX'] == 0
    assert mapping_dict['worldY'] == 0
    assert mapping_dict['worldZ'] == 0


def test_line_intersect_missing_y_axis():
    line1 = Line(pt1=Point2D(x=10, y=0), pt2=Point2D(x=10, y=10), axis=Axis(AxisLabel.X, 0))
    line2 = Line(pt1=Point2D(x=0, y=5), pt2=Point2D(x=30, y=5), axis=Axis(AxisLabel.X, 0))
    intersect = line1.intersect(line2)
    assert intersect is None


def test_line_intersect_missing_x_axis():
    line1 = Line(pt1=Point2D(x=10, y=0), pt2=Point2D(x=10, y=10), axis=Axis(AxisLabel.Y, 0))
    line2 = Line(pt1=Point2D(x=0, y=5), pt2=Point2D(x=30, y=5), axis=Axis(AxisLabel.Y, 0))
    intersect = line1.intersect(line2)
    assert intersect is None
