from slog.geometry_module import *
import numpy as np
import pytest
from math import sqrt


def test_Node():
    """
    Class testing for :class:`Node`
    """
    P1 = Node(2.56, 1)
    P2 = Node(2.56+1e-11, 1+1e-11)
    P3 = Node(2.57, 1)
    P4 = Node(1, 1)
    assert P1.x == 2.56
    assert P1.y == 1
    assert P1 == P2
    assert P1 != P3
    assert P1 + P4 == Node(3.56, 2)
    assert 2 * P1 == P1 * 2 == Node(5.12, 2)
    assert P1.distance_to(P4) == P4.distance_to(P1) == 1.56
    assert P1.on_segment(Segment(Node(0,0), 2 * P1))


def test_Vector():
    """
    Class testing for :class:`Vector`
    """
    u = Vector(1, 1)
    v = Vector(2, -1)
    P0 = Node(0, 0)
    P1 = Node(1, 1)
    w = Vector(P0, P1)
    
    assert v * 2 == 2 * v == Vector(4, -2)
    assert v / 2 == 0.5 * v == Vector(1, -0.5)
    assert u.dot(v) == v.dot(u) == 1
    assert u.cross(v) == -v.cross(u) == -3
    assert -v == Vector(-2, 1)
    assert u + v == Vector(3, 0)
    assert u + v == u - (-v)
    assert P0 + w == P1
    

def test_Segment():
    """
    Class testing for :class:`Segment`
    """
    P0 = Node(0, 1)
    P1 = Node(1, 0)
    P2 = Node(0, 0)
    P3 = Node(1, 1)
    P4 = Node(0.5, 0.5)
    P5 = Node(-0.5, 0.5)
    seg = Segment(P0, P1)
    seg2 = Segment(P2, P3)
    seg3 = Segment(P4, P5)
    seg4 = Segment(P2, P0)
    seg5 = Segment(P3, P4)
    assert seg.vector == Vector(1, -1)
    assert seg._length2() == 2
    assert seg.length == np.sqrt(2)
    assert seg.intersect(seg2) == seg2.intersect(seg) == P4
    assert seg3.intersect(seg4) == seg4.intersect(seg3) == Node(0, 0.5)
    assert seg4.intersect(seg5) is None
    assert seg3.intersect(seg) == P4
    assert seg3.intersect(seg5) == P4


def test_Line():
    """
    Class testing for :class:`Line`
    """
    l = Line([Node(0.75, -0.5), Node(0.75, 1.25), Node(0.25, 0.75), Node(0.25, 0.25)])
    lr = l.reversed()
    assert lr == Line([Node(0.25, 0.25), Node(0.25, 0.75), Node(0.75, 1.25), Node(0.75, -0.5)])
    assert l.dim == lr.dim == 1
    assert l.n == lr.n == 4
    assert l.length == lr.length == 2.25 + sqrt(2) / 2
    P0 = Node(0, 0)
    P1 = Node(1, 0)
    P2 = Node(0.5, 0.5)
    P3 = Node(1, 1)
    P4 = Node(0, 1)
    poly = Polygon([P0, P1, P2, P3, P4])
    assert l.intersect_polygon(poly) == [Line([Node(0.75, 0), Node(0.75, 0.25)]),
                                         Line([Node(0.75, 0.75), Node(0.75, 1)]),
                                         Line([Node(0.5, 1), Node(0.25, 0.75), Node(0.25, 0.25)])]
    #assert lr.intersect_polygon(poly) == [Line([Node(0.25, 0.25), Node(0.25, 0.75), Node(0.5, 1)]),
    #                                     Line([Node(0.75, 1), Node(0.75, 0.75)]),
    #                                     Line([Node(0.75, 0.25), Node(0.75, 0)])]



def test_Polygon():
    """
    Class testing for :class:`Polygon`
    """
    P0 = Node(0, 0)
    P1 = Node(1, 0)
    P2 = Node(0.5, 0.5)
    P3 = Node(1, 1)
    P4 = Node(0, 1)
    PA = Node(0.25, 0.5)
    PB = Node(-0.25, 0.5)
    poly = Polygon([P0, P1, P2, P3, P4])
    assert poly.contains(PA)
    assert PA.in_polygon(poly)
    assert not PB.in_polygon(poly)
    assert not P0.in_polygon(poly)
    assert poly.area == 0.75
    assert poly.perimeter == 3 + np.sqrt(2)
    with pytest.raises(ValueError) as msg:
        polyneg = Polygon([P4, P3, P2, P1, P0])
    with pytest.raises(ValueError) as msg:
        polyinv = Polygon([P0, P1, P4, P3])
    P10 = Node(0, 0)
    P11 = Node(1, 0)
    P12 = Node(1, 1)
    P13 = Node(0, 1)
    P13b = Node(0.25, 0.25)
    P14 = Node(0.5, -0.2)
    P15 = Node(1.2, 0.5)
    P15b = Node(0.75, 0.75)
    P16 = Node(0.5, 1.2)
    P17 = Node(-0.2, 0.5)
    P18 = Node(-1, -1)
    P19 = Node(2, -1)
    P20 = Node(2, 2)
    P21 = Node(-1, 2)
    polyA = Polygon([P10, P11, P12, P13])
    polyB = Polygon([P14, P15, P16, P17])
    assert polyA.intersect(polyB)[0] == Polygon([Node(1, 0.3), Node(1, 0.7),
                                                   Node(0.7, 1), Node(0.3, 1),
                                                   Node(0, 0.7), Node(0, 0.3),
                                                   Node(0.3, 0), Node(0.7, 0)])
    polyC = Polygon([P10, P11, P12, P13, P13b])
    polyD = Polygon([P14, P15, P15b, P16, P17])
    assert polyC.intersect(polyD)[0] == Polygon([Node(1, 0.3), Node(1, 0.61111111111),
                                                  Node(0.75, 0.75),
                                                  Node(0.61111111111, 1), Node(0.3, 1),
                                                  Node(0.075, 0.775), Node(0.25, 0.25),
                                                  Node(0.15, 0.15),
                                                  Node(0.3, 0), Node(0.7, 0)])
    polyE = Polygon([P18, P19, P20, P21])
    assert polyA.intersect(polyE)[0] == polyA
    assert polyE.intersect(polyB)[0] == polyB

    P22 = Node(0.75, -0.5)
    P23 = Node(2, 0)
    P24 = Node(2, 1.5)
    P25 = Node(0.75, 1.5)

    polyF = Polygon([P22, P23, P24, P25])
    polyG = Polygon([Node(0.75, 0), Node(1, 0), Node(0.75, 0.25)])
    polyH = Polygon([Node(0.75, 0.75), Node(1, 1), Node(0.75, 1)])

    assert poly.intersect(polyF) == [polyG, polyH] or poly.intersect(polyF) == [polyH, polyG]