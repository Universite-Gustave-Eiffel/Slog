# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 07:07:49 2021

@author: jadsadek
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.lines as lin 
import matplotlib.colors as col
import slog.intersection_module as inter

# Constants
EPSILON = 1e-10
"""
Tolerance to be used when checking equality of geometrical objects.
"""
INFTY = 1e5
"""
Value to be used to represent infinity.
"""
SCALE = 1
"""
Value to be used to represent infinity.
"""
NUMBERS = (float, int)
"""
Types considered numbers.
"""

# Geometrical classes
class Node(inter.Node):
    r"""
    A class used to represent nodes.


    Parameters
    ----------
    x, y : :class:`float`\ s
        Coordinates of the node
    """    
    def __eq__(self, other):
        if isinstance(other, Node):
            return np.abs(self.x - other.x) < EPSILON and np.abs(self.y - other.y) < EPSILON
        raise TypeError(f"unsupported operand type(s) for ==: 'Node' and '{type(other).__name__}'")

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, vec):
        if isinstance(vec, (Vector, Node)):
            return Node(self.x + vec.x, self.y + vec.y)
        raise TypeError(f"unsupported operand type(s) for +: 'Node' and '{type(vec).__name__}'")    

    def __sub__(self, vec):
        if isinstance(vec, (Vector, Node)):
            return Node(self.x - vec.x, self.y - vec.y)
        raise TypeError(f"unsupported operand type(s) for +: 'Node' and '{type(vec).__name__}'")    

    def __mul__(self, a):
        if isinstance(a, NUMBERS):
            return Node(a * self.x,  a * self.y)
        raise TypeError(f"unsupported operand type(s) for *: 'Node' and '{type(a).__name__}'")
    __rmul__ = __mul__

    def draw(self, ax=None, givenradius=0.03*SCALE, **kwargs):
        r"""
        Draw the Node object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None

        **kwargs : optional keywords
            See :class:`matplotlib.patches.Circle` for a description of available options
        """
        ax, current_figure = _axis_setup(ax)
        p_fill = patch.Circle((self.x, self.y), radius=givenradius, **kwargs)
        ax.add_patch(p_fill)

    def distance_to(self, B):
        r"""
        Calculates the euclidean distance to Node `B`.

        Parameters
        ----------
        B : Node

        Returns
        -------
        float
            Distance to Node `B`
        """
        return  Segment(self, B).length

    def _in_bounding_box(self, seg):
        return (min(seg.A.x, seg.B.x) <= self.x <= max(seg.A.x, seg.B.x) or abs(seg.A.x - seg.B.x)<EPSILON)\
                and (min(seg.A.y, seg.B.y) <= self.y <= max(seg.A.y, seg.B.y) or abs(seg.A.y - seg.B.y)<EPSILON)

    def on_segment(self, seg):
        r"""
        Returns a boolean stating wether the node is on the segment `seg`.

        Parameters
        ----------
        poly : Segment
            The Segment potentially containing the node.

        Returns
        -------
        bool
            True if self is inside the Segment `seg`.
        """
        return self._in_bounding_box(seg) and abs(seg.vector.cross(Vector(seg.A, self))) < EPSILON

    def in_polygon(self, poly):
        r"""
        Returns a boolean stating wether the node is inside the polygon `poly`.

        Parameters
        ----------
        poly : Polygon
            The Polygon potentially containing the node.

        Returns
        -------
        bool
            True if self is inside the polygon `poly`.

        See Also
        --------
        Polygon.contains : reciprocal function.
        """
        return poly.contains(self)


class Vector:
    r"""
    A class representing a vector.

    Parameters
    ----------
    x, y : `float`
        1st and 2nd coordinates of the vector
    """
    def __init__(self, x, y):
        if isinstance(x, NUMBERS) and isinstance(y, NUMBERS):
            self.x = x
            self.y = y
        elif isinstance(x, Node) and isinstance(y, Node):
            self.x = y.x - x.x
            self.y = y.y - x.y
        else:
            TypeError(f"Vector class cannot be created from types {type(x).__name__} and {type(y).__name__}")

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __eq__(self, v):
        if isinstance(v, Vector):
            return np.abs(self.x - v.x) < EPSILON and np.abs(self.y - v.y) < EPSILON
        raise TypeError(f"unsupported operand type(s) for ==: 'Vector' and '{type(v).__name__}'")

    def __add__(self, v):
        if isinstance(v, Vector):
            return Vector(self.x + v.x,  self.y + v.y)
        raise TypeError(f"unsupported operand type(s) for +: 'Vector' and '{type(v).__name__}'")

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __sub__(self, v):
        if isinstance(v, Vector):
            return Vector(self.x - v.x,  self.y - v.y)
        raise TypeError(f"unsupported operand type(s) for -: 'Vector' and '{type(v).__name__}'")
        
    def __mul__(self, a):
        if isinstance(a, NUMBERS):
            return Vector(a * self.x,  a * self.y)
        raise TypeError(f"unsupported operand type(s) for *: 'Vector' and '{type(a).__name__}'")
    __rmul__ = __mul__

    def __truediv__(self, a):
        if isinstance(a, NUMBERS):
            return Vector(self.x/a,  self.y/a)
        raise TypeError(f"unsupported operand type(s) for *: 'Vector' and '{type(a).__name__}'")

    def draw(self, ax=None, originPoint=(0, 0), width=0.002*SCALE, head_width = 0.01*SCALE, linewidth=0, **kwargs):
        r"""
        Draw the Vector object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None

        originPoint : (2,) array_like NUMBER, default=(0, 0)
            Coordinates of the point from which the vector is drawn.

        **kwargs : optional keywords
            See :class:`matplotlib.patches.FancyArrow` for a description of available options
        """
        ax, current_figure = _axis_setup(ax)
        ax.add_patch(patch.FancyArrow(*originPoint, self.x, self.y, width=width, head_width=head_width, linewidth=linewidth, **kwargs))

    def dot(self, v):
        r"""
        Implements the scalar product between `self` and `v`.

        Parameters
        ----------
        v : Vector
            Other vector
        """
        return self.x*v.x + self.y*v.y

    def cross(self, v):
        r"""
        Implements the vector product between `self` and `v`.

        Parameters
        ----------
        v : Vector
            Other vector
        """
        return self.x*v.y - self.y*v.x


class Segment:
    r"""
    A class used to represent segments.

    Parameters
    ----------
    A, B : :class:`Node`\ s
        The two endpoints of the segment.

    Warnings
    --------
    Segments are oriented, segments AB and BA are different.
    This affects the attribute :attr:Segment.vector for instance.
    """
    
    def __init__(self, nodeA, nodeB):
        self.A = nodeA
        self.B = nodeB
        self._display_properties = {'linewidth': SCALE}

    def __eq__(self, other):
        if isinstance(other, Segment):
            return self.A == other.A and self.B == other.B
        raise TypeError(f"unsupported operand type(s) for ==: 'Segment' and '{type(other).__name__}'")

    def __repr__(self):
        return f"Segment({self.A}, {self.B})"

    @property
    def vector(self):
        r"""
        Vector from point A to point B
        """
        return Vector(self.B.x - self.A.x, self.B.y - self.A.y)
    
    @property
    def length(self):
        r"""
        Length of the segment (Euclidean norm).
        """
        return np.sqrt(self._length2())

    def _length2(self):
        return self.vector.dot(self.vector)
    
    def draw(self, ax=None,  **kwargs):
        r"""
        Draw the Segment object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None

        **kwargs : optional keywords
            See :class:`matplotlib.lines.Line2D` for a description of available options
        """
        ax, current_figure = _axis_setup(ax)
        _dic_update_keep_left(kwargs, self._display_properties)
        ax.add_line(lin.Line2D((self.A.x, self.B.x), (self.A.y, self.B.y), **kwargs))

    def intersect(self, other):
        r"""
        Calculates the intersection with another segment.

        Parameters
        ----------
        other : Segment
            Other segment of the intersection.

        Returns
        -------
        Node or None
            Intersection node or `None` if there is no intersection.
        """
        if abs(self.vector.cross(other.vector)) > EPSILON:
            lambd = Segment(self.A, other.A).vector.cross(other.vector)/self.vector.cross(other.vector)
            point = self.A + lambd * self.vector

            return point if point._in_bounding_box(self) and point._in_bounding_box(other) else None
        else:
            # Les segments sont paralleles
            return None

    def contains(self, node):
        return node.on_segment(self)


class Line():
    r"""
    A class used to represent a line.

    Parameters
    ----------
    nodes : list of :class:`Node`
        The nodes composing the line.

    Warnings
    --------
    Lines, like the segmens composing them have an orientation.

    Lines ABC and CBA are considered different.

    Use :meth:`Line.reversed` to change orientation
    """
    
    def __init__(self, nodes):
        if isinstance(nodes, inter.Line):
            self._nodes = list(v.node for v in nodes.vertices())
        else:
            self._nodes = nodes
        self._segments = [Segment(self._nodes[k], self._nodes[(k+1)]) for k in range(self.n-1)]
        #if self.is_selfintersecting():
        #    raise ValueError(f"{type(self).__name__} is self-intersecting.")
        self._display_properties = {'linewidth': SCALE, 'color':'k'}

    @property
    def dim(self):
        r"""
        Dimension of the line = 1
        """
        return 1

    @property
    def n(self):
        r"""
        Number of nodes in the line
        """
        return len(self._nodes)

    @property
    def nodes(self):
        r"""
        Copy of the list of nodes composing the line
        """
        return self._nodes.copy()

    @property
    def segments(self):
        r"""
        Copy of the list of :class:`Segment`\ s composing the line
        """
        return self._segments.copy()

    @property
    def length(self):
        r"""
        Returns the total length of the line.

        Returns
        -------
        float
            Total line length
        """
        s = 0
        for e in self._segments:
            s += e.length
        return s

    def __repr__(self):
        return f"{type(self).__name__}({self.nodes})" 
    
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.n != other.n:
            return False
        for (node1, node2) in zip(self._nodes, other._nodes):
            if node1 != node2:
                return False
        return True

    def __iter__(self):
        return iter(self._nodes)

    def update_display_properties(self, **kwargs):
        self._display_properties.update(kwargs)

    def draw(self, ax=None, **kwargs):
        r"""
        Draw the Line object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None

        **kwargs : optional keywords
            See :class:`matplotlib.lines.Line2D` for a description of available options
        """
        ax, current_figure = _axis_setup(ax)
        _dic_update_keep_left(kwargs, self._display_properties)
        for e in self._segments:
            e.draw(ax, **kwargs)

    def to_nparray(self):
        r"""
        Gives an array with the coordinates of the nodes.

        Used to draw the line

        Returns
        -------
        array
            (2, n) array containing the coordinates of the nodes
        """
        return np.array(list([node.x, node.y] for node in self._nodes))

    def reversed(self):
        r"""
        Returns the line with the opposite direction.

        Returns
        -------
        Line
            The reversed line.
        """
        return Line(self.nodes[::-1])
    
    def is_selfintersecting(self):
        """
        Returns a boolean stating wether the Line is self-intersecting.

        Returns
        -------
        bool
            True if self is self-intersecting.
        """
        for i1 in range(self.n - 2):
            for i2 in range(i1 + 2, self.n - 1 if i1 != 0 else self.n - 2):
                if self._segments[i1].intersect(self._segments[i2]):
                    return True
        return False

    def intersect_polygon(self, poly):
        r"""
        Calculates the intersection with polygon `poly`. Handle degeneracies.

        Parameters
        ----------
        poly : Polygon
            The polgon with which to take the intersection

        Returns
        -------
        list of Line
            The intersection

        Warnings
        --------
        This method has not been thoroughly tested.
        """
        l = []
        for P in inter.intersect_lines_polygons([inter.Line(self)], [inter.Polygon(poly)], EPSILON=EPSILON):
            l.append(Line(P))
        return l 

    def intersect_polygon2(self, poly):
        r"""
        Calculates the intersection with polygon `poly`.

        Parameters
        ----------
        poly : Polygon
            The polgon with which to take the intersection

        Returns
        -------
        list of Line
            The intersection

        Warnings
        --------
        - This method does not cope well with degeneracies such as nodes on edges.

        Please use `intersect_polygon` 
        """
        l1 = self.nodes
        lineList = []
        l3 = []
        lentry = dict()
        insideFlag = l1[0].is_inside(poly)
        for edge1 in self.segments:
            tempNodes = edge1.intersect_line(poly.boundary)
            for node in tempNodes:
                insideFlag = not insideFlag
                lentry[node] = insideFlag
            e1Ai = l1.index(edge1.A)
            l1 = l1[:e1Ai + 1] + tempNodes + l1[e1Ai + 1:]
        
        for (i, candidateNode) in enumerate(l1):
            if candidateNode not in lentry:
                if candidateNode.is_inside(poly):
                    l3.append(candidateNode)
            else:
                l3.append(candidateNode)
                if not lentry[candidateNode]:   
                    lineList.append(Line(l3)) 
                    l3=[]
        if l3:
            lineList.append(Line(l3))
        return lineList     

    def contains(self, node):
        return any(node.on_segment(seg) for seg in self.segments)


class Polygon():
    r"""
    A class used to represent a polygon.


    Parameters
    ----------
    nodes : list of Node
        Ordered list of nodes composing the polygon.

    Warnings
    --------
    Nodes should be given in an anticlockwise fasion in order to ensure positive area.
    """
    
    def __init__(self, nodes):
        if isinstance(nodes, inter.Polygon):
            self._nodes = list(v.node for v in nodes.vertices())
            if self.area < 0:
                self._nodes = self._nodes[::-1]
        else:
            self._nodes = list(nodes)
            if self.area < 0:
                raise ValueError(f"Given nodes define a negative-area {type(self).__name__}.")
        self._segments = [Segment(self._nodes[k], self._nodes[(k+1) % self.n]) for k in range(self.n)]
        if self.is_selfintersecting():
            raise ValueError(f"{type(self).__name__} is self-intersecting.")
        self._display_properties = {'linewidth': SCALE, 'fill': False}

    @property
    def dim(self):
        r"""
        Dimension of the Polygon object = 2
        """
        return 1

    @property
    def n(self):
        r"""
        Number of nodes in the polygon, also the number of edges.
        """
        return len(self._nodes)

    @property
    def nodes(self):
        r"""
        Copy of the list of nodes.
        """
        return self._nodes.copy()
    
    @property
    def segments(self):
        r"""
        Copy of the list of segments.
        """
        return self._segments.copy()

    @property
    def perimeter(self):
        r"""
        Returns the perimeter of the polygon.

        Returns
        -------
        float
            Perimeter of the polygon
        """
        s = 0
        for e in self.segments:
            s += e.length
        return s

    @property
    def area(self):
        r"""
        Returns the area of the polygon.

        Returns
        -------
        float
            Area of the polygon
        """
        return sum([Vector(self._nodes[0], self._nodes[i]).cross(Vector(self._nodes[0], self._nodes[i+1])) for i in range(1, self.n-1)])/2

    @property
    def boundary(self):
        return Line(self.nodes + self._nodes[:1])

    def __repr__(self):
        return f"{type(self).__name__}({self.nodes})" 
    
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.n != other.n:
            return False
        for (initial, node) in enumerate(other.nodes):
            if node == self.nodes[0]:
                break
        for k in range(initial, self.n + initial):
            if self.nodes[k - initial] != other.nodes[k % self.n]:
                return False
        return True

    def __iter__(self):
        return iter(self._nodes)

    def update_display_properties(self, **kwargs):
        self._display_properties.update(kwargs)

    def draw(self, ax=None, **kwargs):
        r"""
        Draw the Polygon object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None

        **kwargs : optional keywords
            See :class:`matplotlib.patches.Polygon` for a description of available options
        """
        ax, current_figure = _axis_setup(ax)
        _dic_update_keep_left(kwargs, self._display_properties)
        self._patch = ax.add_patch(patch.Polygon(self.to_nparray(), **kwargs))

    def is_selfintersecting(self):
        """
        Returns a boolean stating wether the Polygon is self-intersecting.

        Returns
        -------
        bool
            True if self is self-intersecting.
        """
        for i1 in range(self.n - 1):
            for i2 in range(i1 + 2, self.n if i1 != 0 else self.n - 1):
                if self._segments[i1].intersect(self._segments[i2]):
                    return True
        return False

    def contains(self, node):
        """
        Returns a boolean stating wether the Polygon contains the Node node.

        Parameters
        ----------
        node : Node
            The candidate node.

        Returns
        -------
        bool
            True if self contains node.

        See Also
        --------
        Node.is_inside : reciprocal function.
        """
        return inter.Polygon(self).pointInPoly(node)

    def contains2(self, node):
        """
        Returns a boolean stating wether the Polygon contains the Node node.

        Parameters
        ----------
        node : Node
            The candidate node.

        Returns
        -------
        bool
            True if self contains node.

        See Also
        --------
        Node.is_inside : reciprocal function.
        """
        if self.boundary.contains(node):
            return True
        P = Node(INFTY, 3.215)
        seg = Segment(node, P)
        inside = False
        for edge in self.segments:
            if seg.intersect(edge) is not None:
                inside = not inside
        return inside

    def intersect(self, poly):
        r"""
        Calculates the intersection with another polygon. Handles degenerecies.

        Parameters
        ----------
        other : Polygon
            The polgon with which to take the intersection

        Returns
        -------
        list of Polygon
            The intersection

        Warnings
        --------
        This method has not been thoroughly tested. Discrepencies may exist with the original C++ algorithm. 
        
        Notes
        -----
        .. note::

            This method is based on the revised Greiner-Hormann-Foster algorithm.

            It has been translated from `Hormann's implementation in C++ <https://www.inf.usi.ch/hormann/polyclip/>`_.

            It uses the class :class:`intersection_module.VPolygon` and the function `intersection_module.intersect_vpolygons` from the module :ref:`intersection_module`.
        """
        l = []
        for P in inter.intersect_polygons([inter.Polygon(self)], [inter.Polygon(poly)], EPSILON=EPSILON):
            l.append(Polygon(P))
        return l

    def intersect2(self, poly):
        r"""
        Calculates the intersection with another polygon. Does not handle degeneracies.

        This method is based on the original Greiner-Hormann algorithm.

        Parameters
        ----------
        other : Polygon
            The polgon with which to take the intersection

        Returns
        -------
        list of Polygon
            The intersection

        Warnings
        --------
        - This method does not cope well with degeneracies such as polygons with aligned nodes, nodes on edges, edges on edges, etc.

        Please use `intersect` 
        """
        l1 = self.nodes
        l2 = poly.nodes
        l3 = []
        intersectionNodes = [list() for _ in poly.segments]
        lentry = dict()
        insideFlag = l1[0].is_inside(poly)
        for edge1 in self.segments:
            tempNodes = []
            for (i, edge2) in enumerate(poly.segments):
                intersectionPoint = edge1.intersect(edge2)
                if intersectionPoint is not None:
                    tempNodes.append(intersectionPoint)
                    intersectionNodes[i].append(intersectionPoint)
            tempNodes.sort(key=edge1.A.distance_to)
            for node in tempNodes:
                insideFlag = not insideFlag
                lentry[node] = insideFlag
            e1Ai = l1.index(edge1.A)

            l1 = l1[:e1Ai + 1] + tempNodes + l1[e1Ai + 1:]
        for (edge2, lnodes) in zip(poly.segments, intersectionNodes):
            lnodes.sort(key=edge2.A.distance_to)
            e2Ai = l2.index(edge2.A)
            l2 = l2[:e2Ai + 1] + lnodes + l2[e2Ai + 1:]
        
        for (i, candidateNode) in enumerate(l1):
            if candidateNode in lentry:
                break
        else:
            if insideFlag:
                return [self]
            elif l2[0].is_inside(self):
                return [poly]
            return []
        L = [l2, l1]
        current_list = False
        polygonList = []
        firstNode = candidateNode
        while True:
            l3.append(candidateNode) 
            if candidateNode in lentry:
                current_list = lentry.pop(candidateNode)
            candidateNode = L[current_list][(L[current_list].index(candidateNode) + 1) % len(L[current_list])]
            if candidateNode == firstNode:
                polygonList.append(Polygon(l3))
                if lentry:
                    candidateNode = next(iter(lentry))
                    firstNode = candidateNode
                    l3 = []
                else:
                    break
        return polygonList     

    def to_nparray(self):
        r"""
        Gives an array with the coordinates of the nodes.

        Used to draw the polygon

        Returns
        -------
        array
            (2, n) array containing the coordinates of the nodes
        """
        return np.array(list([node.x, node.y] for node in self._nodes))


def _axis_setup(ax=None):
    #inspired by networkx.graph
    if ax is None:
        current_figure = plt.gcf()
    else:
        current_figure = ax.get_figure()
    if ax is None:
        if current_figure._axstack() is None:
            ax = current_figure.add_axes((0, 0, 1, 1))
        else:
            ax = current_figure.gca()
    return (ax, current_figure)

def _dic_update_keep_left(dic1, dic2):
    for key in dic2:
        if key not in dic1:
            dic1[key] = dic2[key]


if __name__ == "__main__":
    P0 = Node(0, 0)
    P1 = Node(1, 0)
    P2 = Node(1, 0.8)
    P3 = Node(0.7, 0.8)
    P4 = Node(0.7, 0.5)
    P5 = Node(0.3, 0.5)
    P6 = Node(0.3, 0.8)
    P7 = Node(0, 0.8)
    geom = ([P0, P1, P2, P3, P4, P5, P6, P7])
 
    Q1 = Node(-1, -1)
    Q2 = Node(2, -1)
    Q3 = Node(2, 1)
    Q4 = Node(0.5, 0.55)
    Q5 = Node(-1, 1)
    Q2b = Node(2, 2)
    Q1b = Node(-1, 2)
    f1 = Polygon([Q1, Q2, Q3, Q4, Q5])
    f2 = Polygon([Q1b, Q5, Q4, Q3, Q2b])
    f1.draw(fill=False)
    f2.draw(fill=False)
    geom.intersect(f1)[0].draw(fc='g')
    geom.intersect(f2)[0].draw(fc='b')
    l = Line([Node(0.75, -0.5), Node(0.75, 1.25), Node(0.25, 0.75), Node(0.25, 0.25)])
    lr = l.reversed()
    l.draw()
    plt.show()



