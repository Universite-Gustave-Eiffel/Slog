from enum import Enum



class IntersectionLabel(Enum):
    NONE = 0
    CROSSING = 1
    BOUNCING = 2
    LEFT_ON = 3
    RIGHT_ON = 4
    ON_ON = 5
    ON_LEFT = 6
    ON_RIGHT = 7
    DELAYED_CROSSING = 8
    DELAYED_BOUNCING = 9


class EntryExitLabel(Enum):
    EXIT = 0   
    ENTRY = 1
    NEITHER = 2


class IteratorType(Enum):
    SOURCE = 0
    INTERSECTION = 1
    CROSSING_INTERSECTION = 2
    ALL = 3


class IntersectionType(Enum):
    NO_INTERSECTION = 0
    X_INTERSECTION = 1
    T_INTERSECTION_Q = 2
    T_INTERSECTION_P = 3
    V_INTERSECTION = 4
    X_OVERLAP = 5
    T_OVERLAP_Q = 6
    T_OVERLAP_P = 7
    V_OVERLAP = 8


class RelativePositionType(Enum):
    LEFT = 0
    RIGHT = 1
    IS_P_m = 2
    IS_P_p = 3


def toggle(status):
    if status is EntryExitLabel.ENTRY:
        return EntryExitLabel.EXIT
    elif status is EntryExitLabel.EXIT:
        return EntryExitLabel.ENTRY
    return status



class Node:
    r"""
    A class used to represent nodes.


    Parameters
    ----------
    x, y : :class:`float`\ s
        Coordinates of the node
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Node({self.x}, {self.y})"

    def __add__(self, vec):
        return Node(self.x + vec.x, self.y + vec.y)
        
    def __sub__(self, vec):
        return Node(self.x - vec.x, self.y - vec.y)

    def __mul__(self, a):
        return Node(a * self.x,  a * self.y)
    __rmul__ = __mul__

def A(P, Q, R):
    return (Q.x-P.x) * (R.y-P.y) - (Q.y-P.y) * (R.x-P.x)

def dot(P, Q):
    return (Q.x*P.x) + (Q.y*P.y)


class Vertex():
    def __init__(self, node, alpha=-1):
        if isinstance(node, Vertex):
            raise ValueError
        self.node = node
        self.alpha = alpha
        self.prev = None
        self.next = None
        self.neighbour = None
        self.source = False
        self.intersection = False
        self.label = IntersectionLabel.NONE
        self.enex = EntryExitLabel.NEITHER
        self.type = None
    def __repr__(self):
        return "Vertex" + repr(self.node)[4:]

def link(P, Q):
    P.neighbour = Q
    Q.neighbour = P
    P.intersection = True
    Q.intersection = True

def insertVertex(V, curr, alpha=-1):
    if isinstance(curr, Edge):
        insertVertex(V, curr.one, V.alpha)
    elif isinstance(curr, Vertex):
        if alpha > -1.0:
            curr = curr.next
            while (not curr.source and curr.alpha < alpha):
                curr = curr.next
        else:
            curr = curr.next

        curr.prev.next = V
        V.prev = curr.prev
        V.next = curr
        curr.prev = V

class VertexIterator:
    class Iterator:
        def __init__(self, root, iterType = IteratorType.ALL):
            self.root = root
            self.V = None
            self.iterType = iterType
        
        def __next__(self):
            self.nextVertex()
            return self.V

        def __neq__ (self, other):
            return self.root != other.root or self.V != other.V

        def nextVertex(self):
            nextFound = False
            if self.V is None:
                self.V = self.root
                if self.iterType is IteratorType.ALL:
                    nextFound = True
                elif self.iterType is IteratorType.SOURCE:
                    if self.V.source:
                        nextFound = True
                elif self.iterType is IteratorType.INTERSECTION:
                    if self.V.intersection:
                        nextFound = True
                elif self.iterType is IteratorType.CROSSING_INTERSECTION:
                    if self.V.intersection and self.V.label is IntersectionLabel.CROSSING:
                        nextFound = True

            while not nextFound:
                self.V = self.V.next
                if self.iterType is IteratorType.ALL:
                    pass
                elif self.iterType is IteratorType.SOURCE:
                    while (self.V is not self.root and self.V is not None) and not self.V.source:
                        self.V = self.V.next
                elif self.iterType is IteratorType.INTERSECTION:
                    while (self.V is not self.root and self.V is not None) and (not self.V.intersection):
                        self.V = self.V.next
                elif self.iterType is IteratorType.CROSSING_INTERSECTION:
                    while (self.V is not self.root and self.V is not None) and ((not self.V.intersection) or (self.V.label is not IntersectionLabel.CROSSING)):
                        self.V = self.V.next
                
                if self.V is self.root or self.V is None:
                    self.V = None
                    self.root = None
                    raise StopIteration

                if self.iterType is IteratorType.ALL:
                    nextFound = True
                elif self.iterType is IteratorType.SOURCE:
                    if self.V.source:
                        nextFound = True
                elif self.iterType is IteratorType.INTERSECTION:
                    if self.V.intersection:
                        nextFound = True
                elif self.iterType is IteratorType.CROSSING_INTERSECTION:
                    if self.V.intersection and self.V.label is IntersectionLabel.CROSSING:
                        nextFound = True
            return self.V

    def __init__(self):
        self.root = None

    def __iter__(self):
        return self.Iterator(self.root, self.iterType)

    #def end(self):
    #    return self.Iterator(None, self.iterType)


class Edge:
    def __init__(self, P, Q):
        self.one = P
        self.two = Q

    def __repr__(self):
        return "Edge(" + repr(self.one) + ", " + repr(self.two) +")"

class EdgeIterator:
    class Iterator:
        def __init__(self, root, iterType):
            self.root = root
            self.one = None
            self.two = None
            self.iterType = iterType

        def __neq__ (self, other):
            return self.root != other.root or self.one != other.one or self.two != other.two

        def nextVertex(self, curr):
            if curr is None:
                raise StopIteration
            curr = curr.next
            if curr is None:
                raise StopIteration
            if self.iterType is IteratorType.ALL:
                pass
            elif self.iterType is IteratorType.SOURCE:
                while not curr.source:
                    curr = curr.next
            return curr

        def nextEdge(self):
            if self.root is None:
                raise StopIteration
            if self.one is None:
                self.one = self.root
                self.two = self.nextVertex(self.one)
                if self.two is self.one:
                    raise StopIteration
                return self.one
            if self.two is self.root or self.two is None:
                self.root = None
                self.one = None
                self.two = None
                raise StopIteration
            self.one = self.two
            self.two = self.nextVertex(self.one)
            return self.one

        def __next__(self):
            self.nextEdge()
            return Edge(self.one, self.two)
    
    def __init__(self):
        self.root = None

    def __iter__(self):
        return self.Iterator(self.root, self.iterType)


class Polygon:
    def __init__(self, nodes=None):
        self.root = None
        self.vertexIter = VertexIterator()
        self.edgeIter = EdgeIterator()
        if nodes is not None:
            for node in nodes:
                self.newVertex(node, True)
    
    def newVertex(self, node, source=False):
        if  isinstance(node, Vertex):
            raise TypeError
        V = Vertex(node)
        V.source = source

        if self.root is None:
            V.next = V
            V.prev = V
            self.root = V
        else:
            V.prev = self.root.prev
            V.next = self.root
            self.root.prev.next = V
            self.root.prev = V

    def removeVertex(self, V):
        if self.root is V:
            self.root = V.next
            if self.root.next is self.root:
                self.root = None
        V.prev.next = V.next
        V.next.prev = V.prev

    def pointInPoly(self, R):
        w = 0
        for E in self.edges(IteratorType.ALL):
            P0 = E.one.node
            P1 = E.two.node

            if (P0.y < R.y) != (P1.y < R.y):
                if P0.x >= R.x:
                    if P1.x >= R.x:
                        w = w + 2 * (P1.y > P0.y) - 1
                    else:
                        if (A(P0, P1, R)>0) == (P1.y > P0.y):
                            w = w + 2 * (P1.y > P0.y) - 1
                else:
                    if (P1.x > R.x):
                        if (A(P0,P1,R) > 0) == (P1.y > P0.y):
                            w = w + 2 * (P1.y > P0.y) - 1
        return (w % 2) != 0

    def allOnOn(self):
        for V in self.vertices(IteratorType.ALL):
            if V.label is not IntersectionLabel.ON_ON:
                return False
        return True

    def noCrossingVertex(self, union_case = False):
        for V in self.vertices(IteratorType.ALL):
            if V.intersection:
                if (V.label is IntersectionLabel.CROSSING) or (V.label is IntersectionLabel.DELAYED_CROSSING):
                    return False
                if union_case and (V.label is IntersectionLabel.BOUNCING or V.label is IntersectionLabel.DELAYED_BOUNCING):
                    return False
        return True

    def getNonIntersectionPoint(self):
        for V in self.vertices(IteratorType.ALL):
            if not V.intersection:
                return V.node

        for V in self.vertices(IteratorType.ALL):
            if (V.next.neighbour is not V.neighbour.prev) and (V.next.neighbour is not V.neighbour.next):
                return 0.5 * (V.node + V.next.node)

    def getNonIntersectionVertex(self):
        for V in self.vertices(IteratorType.ALL):
            if not V.intersection:
                return V

        for V in self.vertices(IteratorType.ALL):
            if (V.next.neighbour is not V.neighbour.prev) and (V.next.neighbour is not V.neighbour.next):
                p = 0.5 * (V.node + V.next.node)
                T = Vertex(p)
                insertVertex(T, V)
                return T
        return None

    def vertices(self, iterType=IteratorType.ALL, first=None):
        self.vertexIter.iterType = iterType

        if first is None:
            self.vertexIter.root = self.root
        else:
            self.vertexIter.root = first
        return self.vertexIter

    def edges(self, iterType=IteratorType.ALL):
        self.edgeIter.iterType = iterType
        self.edgeIter.root = self.root
        return self.edgeIter

    def __repr__(self):
        if self.root is None:
            return "Polygon()"
        s = ""
        for V in self.vertices(IteratorType.ALL):
            s += str(V.node) + ", "
        return "VPolygon(" + s[:-2] + ")"
      


class Line:
    def __init__(self, nodes=None):
        self.root = None
        self.end = None
        self.vertexIter = VertexIterator()
        self.edgeIter = EdgeIterator()
        if nodes is not None:
            for node in nodes:
                self.newVertex(node, True)
    
    def newVertex(self, node, source=False):
        if  isinstance(node, Vertex):
            raise TypeError
        V = Vertex(node)
        V.source = source
        V.next = None
        V.prev = self.end

        if self.end is None:  
            self.root = V
            self.end = V
        else:
            self.end.next = V
            self.end = V
            

    def removeVertex(self, V):
        if self.root is V and self.end is V:
            self.root = V.next
            self.end = V.prev
        elif self.root is V:
            self.root = V.next
            V.next.prev = V.prev
        elif self.end is V:
            self.end = V.prev
            V.prev.next = V.next
        else:
            V.prev.next = V.next
            V.next.prev = V.prev

    def allOnOn(self):
        for V in self.vertices(IteratorType.ALL):
            if V.label is not IntersectionLabel.ON_ON:
                return False
        return True

    def noCrossingVertex(self, union_case = False):
        for V in self.vertices(IteratorType.ALL):
            if V.intersection:
                if (V.label is IntersectionLabel.CROSSING) or (V.label is IntersectionLabel.DELAYED_CROSSING):
                    return False
                if union_case and (V.label is IntersectionLabel.BOUNCING or V.label is IntersectionLabel.DELAYED_BOUNCING):
                    return False
        return True

    def getNonIntersectionPoint(self):
        for V in self.vertices(IteratorType.ALL):
            if not V.intersection:
                return V.node

        for V in self.vertices(IteratorType.ALL):
            if (V.next.neighbour is not V.neighbour.prev) and (V.next.neighbour is not V.neighbour.next):
                return 0.5 * (V.node + V.next.node)

    def getNonIntersectionVertex(self):
        for V in self.vertices(IteratorType.ALL):
            if not V.intersection:
                return V

        for V in self.vertices(IteratorType.ALL):
            if (V.next.neighbour is not V.neighbour.prev) and (V.next.neighbour is not V.neighbour.next):
                p = 0.5 * (V.node + V.next.node)
                T = Vertex(p)
                insertVertex(T, V)
                return T
        return None

    def vertices(self, iterType=IteratorType.ALL, first=None):
        self.vertexIter.iterType = iterType

        if first is None:
            self.vertexIter.root = self.root
        else:
            self.vertexIter.root = first
        self.vertexIter.end = self.end
        return self.vertexIter

    def edges(self, iterType=IteratorType.ALL):
        self.edgeIter.iterType = iterType
        self.edgeIter.root = self.root
        self.edgeIter.end = self.end
        return self.edgeIter

    def __repr__(self):
        if self.root is None:
            return "Polygon()"
        s = ""
        for V in self.vertices(IteratorType.ALL):
            s += str(V.node) + ", "
        return "VPolygon(" + s[:-2] + ")"



def intersect(edgeP, edgeQ, EPSILON):
    P1 = edgeP.one.node
    P2 = edgeP.two.node
    Q1 = edgeQ.one.node
    Q2 = edgeQ.two.node

    AP1 = A(P1, Q1, Q2)
    AP2 = A(P2, Q1, Q2)

    if abs(AP1 - AP2) > EPSILON:
        AQ1 = A(Q1,P1,P2)
        AQ2 = A(Q2,P1,P2)
        alpha = AP1 / (AP1-AP2)
        beta  = AQ1 / (AQ1-AQ2)
        alpha_is_0 = False
        alpha_in_0_1 = False

        if (alpha > EPSILON) and (alpha < 1.0-EPSILON):
            alpha_in_0_1 = True
        else:
            if abs(alpha) <= EPSILON:
                alpha_is_0 = True

        beta_is_0 = False
        beta_in_0_1 = False

        if (beta > EPSILON) and (beta < 1.0-EPSILON):
            beta_in_0_1 = True
        else:
            if abs(beta) <= EPSILON:
                beta_is_0 = True

        if alpha_in_0_1 and beta_in_0_1:
            return IntersectionType.X_INTERSECTION, alpha, beta

        if alpha_is_0 and beta_in_0_1:
            return IntersectionType.T_INTERSECTION_Q, alpha, beta

        if beta_is_0 and alpha_in_0_1:
            return IntersectionType.T_INTERSECTION_P, alpha, beta

        if alpha_is_0 and beta_is_0:
            return IntersectionType.V_INTERSECTION, alpha, beta
    else:
        if abs(AP1) < EPSILON:
            dP = P2 - P1
            dQ = Q2 - Q1
            PQ = Q1 - P1

            alpha = dot(PQ, dP) / dot(dP, dP)
            beta = -dot(PQ, dQ) / dot(dQ, dQ)

            alpha_is_0 = False
            alpha_in_0_1 = False
            alpha_not_in_0_1 = False

            if (alpha > EPSILON) and (alpha < 1.0-EPSILON):
                alpha_in_0_1 = True
            else:
                if abs(alpha) <= EPSILON:
                    alpha_is_0 = True
                else:
                    alpha_not_in_0_1 = True

            beta_is_0 = False
            beta_in_0_1 = False
            beta_not_in_0_1 = False

            if (beta > EPSILON) and (beta < 1.0-EPSILON):
                beta_in_0_1 = True
            else:
                if abs(beta) <= EPSILON:
                    beta_is_0 = True
                else:
                    beta_not_in_0_1 = True

            if alpha_in_0_1 and beta_in_0_1:
                return IntersectionType.X_OVERLAP, alpha, beta

            if alpha_not_in_0_1 and beta_in_0_1:
                return IntersectionType.T_OVERLAP_Q, alpha, beta

            if beta_not_in_0_1 and alpha_in_0_1:
                return IntersectionType.T_OVERLAP_P, alpha, beta

            if alpha_is_0 and beta_is_0:
                return IntersectionType.V_OVERLAP, alpha, beta
    return IntersectionType.NO_INTERSECTION, 0, 0

def computeIntersections(PP, QQ, EPSILON, VERBOSE = False):
    if VERBOSE:
        print("Computing intersections...")
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for P in PP:
        for edgeP in P.edges(IteratorType.SOURCE):
            for Q in QQ:
                for edgeQ in Q.edges(IteratorType.SOURCE):
                    i, alpha, beta = intersect(edgeP, edgeQ, EPSILON=EPSILON)
                    count[i.value] +=1
                    P1 = edgeP.one
                    Q1 = edgeQ.one
                    if i is IntersectionType.X_INTERSECTION:
                        I = (1.0-alpha)*edgeP.one.node + alpha*edgeP.two.node
                        I_P = Vertex(I, alpha)
                        I_Q = Vertex(I, beta)
                        insertVertex(I_P, edgeP)
                        insertVertex(I_Q, edgeQ)
                        link(I_P, I_Q)
                    elif i is IntersectionType.X_OVERLAP:
                        I_Q = Vertex(P1.node, beta)
                        insertVertex(I_Q, edgeQ)
                        link(P1, I_Q)
                        I_P = Vertex(Q1.node, alpha)
                        insertVertex(I_P, edgeP)
                        link(I_P, Q1)
                    elif i is IntersectionType.T_INTERSECTION_Q or i is IntersectionType.T_OVERLAP_Q:
                        I_Q = Vertex(P1.node, beta)
                        insertVertex(I_Q, edgeQ)
                        link(P1, I_Q)
                    elif i is IntersectionType.T_INTERSECTION_P or i is IntersectionType.T_OVERLAP_P:
                        I_P = Vertex(Q1.node, alpha)
                        insertVertex(I_P, edgeP)
                        link(I_P, Q1)
                    elif i is IntersectionType.V_INTERSECTION or i is IntersectionType.V_OVERLAP:
                        link(P1, Q1)
    if VERBOSE:
        print(f"{count[1]} non-degenerate and {count[2]+count[3]+count[4]+count[5]+count[6]+count[7]+count[8]} degenerate intersections found")
        print(f"{count[2] + count[3]} T-intersections")
        print(f"{count[4]} V-intersections")
        print(f"{count[5]} X-overlaps")
        print(f"{count[6] + count[7]} T-overlaps")
        print(f"{count[8]} V-overlaps")
        print(f"...  {count[1] + count[5] + count[3] + count[7]} vertices added to P")
        print(f"...  {count[1] + count[5] + count[2] + count[6]} vertices added to Q")

def oracle(Q, P1, P2, P3):
    if P1.intersection and P1.neighbour is Q:
        return RelativePositionType.IS_P_m
    if P3.intersection and P3.neighbour is Q:
        return RelativePositionType.IS_P_p
    s1 = A(Q.node, P1.node, P2.node)
    s2 = A(Q.node, P2.node, P3.node)
    s3 = A(P1.node, P2.node, P3.node)
    if s3 > 0:
        if s1 > 0 and s2 > 0:
            return RelativePositionType.LEFT
        else:
            return RelativePositionType.RIGHT
    else:
        if s1 < 0 and s2 < 0:
            return RelativePositionType.RIGHT
        else:
            return RelativePositionType.LEFT

def labelIntersections(PP, QQ, RR, UNION = False, VERBOSE = False):
    if VERBOSE:
        print("Labelling intersections...")

    # 1) initial classification
    count = [0,0]
    for P in PP:
        for I in P.vertices(IteratorType.INTERSECTION):
            P_m = I.prev
            P_p = I.next
            Q_m = I.neighbour.prev
            Q_p = I.neighbour.next
            Q_m_type = oracle(Q_m, P_m, I, P_p)
            Q_p_type = oracle(Q_p, P_m, I, P_p)
            if (Q_m_type is RelativePositionType.LEFT and Q_p_type is RelativePositionType.RIGHT) \
                or (Q_m_type is RelativePositionType.RIGHT and Q_p_type is RelativePositionType.LEFT):
                I.label = IntersectionLabel.CROSSING
                count[0] += 1
            if (Q_m_type is RelativePositionType.LEFT and Q_p_type is RelativePositionType.LEFT) \
                or (Q_m_type is RelativePositionType.RIGHT and Q_p_type is RelativePositionType.RIGHT):
                I.label = IntersectionLabel.BOUNCING
                count[1] += 1
            
            if (Q_p_type is RelativePositionType.IS_P_p and Q_m_type is RelativePositionType.RIGHT) \
                or (Q_m_type is RelativePositionType.IS_P_p and Q_p_type is RelativePositionType.RIGHT):
                I.label = IntersectionLabel.LEFT_ON
            if (Q_p_type is RelativePositionType.IS_P_p and Q_m_type is RelativePositionType.LEFT) \
                or (Q_m_type is RelativePositionType.IS_P_p and Q_p_type is RelativePositionType.LEFT):
                I.label = IntersectionLabel.RIGHT_ON
            if (Q_p_type is RelativePositionType.IS_P_p and Q_m_type is RelativePositionType.IS_P_m) \
                or (Q_m_type is RelativePositionType.IS_P_p and Q_p_type is RelativePositionType.IS_P_m):
                I.label = IntersectionLabel.ON_ON
            if (Q_m_type is RelativePositionType.IS_P_m and Q_p_type is RelativePositionType.RIGHT) \
                or (Q_p_type is RelativePositionType.IS_P_m and Q_m_type is RelativePositionType.RIGHT):
                I.label = IntersectionLabel.ON_LEFT
            if (Q_m_type is RelativePositionType.IS_P_m and Q_p_type is RelativePositionType.LEFT) \
                or (Q_p_type is RelativePositionType.IS_P_m and Q_m_type is RelativePositionType.LEFT):
                I.label = IntersectionLabel.ON_RIGHT        
    if VERBOSE:
        print(f"... {count[0]} crossing and {count[1]} bouncing intersection vertices")

    # 2) classify intersection chains
    count[0] = 0
    count[1] = 0
    for P in PP:
        for I in P.vertices(IteratorType.INTERSECTION):
            if (I.label is IntersectionLabel.LEFT_ON) or (I.label is IntersectionLabel.RIGHT_ON):
                if I.label is IntersectionLabel.LEFT_ON:
                    x = RelativePositionType.LEFT
                else:
                    x = RelativePositionType.RIGHT
                X = I
                I.label = IntersectionLabel.NONE
                I = I.next
                while I.label is IntersectionLabel.ON_ON:
                    I.label = IntersectionLabel.NONE
                    I = I.next
                if I.label is IntersectionLabel.ON_LEFT:
                    y = RelativePositionType.LEFT
                else:
                    y = RelativePositionType.RIGHT
                if x is not y:
                    chainType = IntersectionLabel.DELAYED_CROSSING
                    count[0] += 1
                else:
                    chainType = IntersectionLabel.DELAYED_BOUNCING
                    count[1] += 1
                X.label = chainType
                I.label = chainType
    if VERBOSE:
        print(f"... {count[0]} delayed crossings and {count[1]} delayed bouncings")

    # 3) copy labels from P to Q
    for P in PP:
        for I in P.vertices(IteratorType.INTERSECTION):
            I.neighbour.label = I.label  
            
    # 3.5) check for special cases
    noIntersection = [[], []]
    identical = [[], []]
    count[0] = 0
    count[1] = 0
    for i in range(2):
        if i == 0:
            P_or_Q = PP
            Q_or_P = QQ
        elif i == 1:
            P_or_Q = QQ
            Q_or_P = PP
        for P in P_or_Q:
            if P.noCrossingVertex(UNION):
                noIntersection[i].append(P)
                if P.allOnOn():
                    identical[i].append(P)
                else:
                    isInside = False
                    p = P.getNonIntersectionPoint()
                    for Q in Q_or_P:
                        if Q.pointInPoly(p):
                            isInside = not isInside
                    if isInside != UNION:
                        RR.append(P)
                        count[0] += 1
    
    for P in identical[0]:
        P_isHole = False
        for P_ in PP:
            if (P_.root is not P.root) and P_.pointInPoly(P.root.node):
                P_isHole = not P_isHole
        for Q in identical[1]:
            breakFlag = False
            for V in Q.vertices(IteratorType.ALL):
                if V is P.root.neighbour:
                    Q_isHole = False
                    for Q_ in QQ:
                        if (Q_.root is not Q.root) and Q_.pointInPoly(Q.root.node):
                            Q_isHole = not Q_isHole
                    if P_isHole != Q_isHole:
                        RR.append(P.operatorStar())
                        count[i] += 1
                    breakFlag = True 
                    break
            if breakFlag:
                break
    if VERBOSE:
        print(f"... {count[0]} interior and {count[1]} identical components added to result")

    # 4) set entry/exit flags
    split = [[], []]
    crossing = [[], []]

    for i in range(2):
        if i == 0:
            P_or_Q = PP
            Q_or_P = QQ
        elif i == 1:
            P_or_Q = QQ
            Q_or_P = PP
        for P in P_or_Q:
            if (P in noIntersection[i]):
                continue
            V = P.getNonIntersectionVertex()
            status = EntryExitLabel.ENTRY
            for Q in Q_or_P:
                if Q.pointInPoly(V.node):
                    status = toggle(status)
            first_chain_vertex = True
            for I in P.vertices(IteratorType.INTERSECTION, V):
                if I.label is IntersectionLabel.CROSSING:
                    I.enex = status
                    status = toggle(status)
                if I.label is IntersectionLabel.BOUNCING and ((status is EntryExitLabel.EXIT) != UNION):
                    split[i].append(I)
                if I.label is IntersectionLabel.DELAYED_CROSSING:
                    I.enex = status
                    if first_chain_vertex:
                        if (status is EntryExitLabel.EXIT) != UNION:
                            I.label = IntersectionLabel.CROSSING
                        first_chain_vertex = False
                    else:
                        if (status is EntryExitLabel.ENTRY) != UNION:
                            I.label = IntersectionLabel.CROSSING
                        first_chain_vertex = True
                        status = toggle(status)
                if I.label is IntersectionLabel.DELAYED_BOUNCING:
                    I.enex = status
                    if first_chain_vertex:
                        if (status is EntryExitLabel.EXIT) != UNION:
                            crossing[i].append(I)
                        first_chain_vertex = False
                    else:
                        if (status is EntryExitLabel.ENTRY) != UNION:
                            crossing[i].append(I)
                        first_chain_vertex = True
                    status = toggle(status)
    # 5) handle split vertex pairs
    count[0] = 0
    for I_P in split[0]:
        I_Q = I_P.neighbour
        if I_Q in split[1]:
            count[0] += 1
            V_P = Vertex(I_P.node)
            V_Q = Vertex(I_Q.node)

            sP = A(I_P.prev.node, I_P.node, I_P.next.node)
            sQ = A(I_Q.prev.node, I_Q.node, I_Q.next.node)
            if sP * sQ > 0:
                link(I_P, V_Q)
                link(I_Q, V_P)
            else:
                link(V_P, V_Q)
            insertVertex(V_P, I_P)
            insertVertex(V_Q, I_Q)

            if not UNION:
                I_P.enex = EntryExitLabel.EXIT
                V_P.enex = EntryExitLabel.ENTRY
                I_Q.enex = EntryExitLabel.EXIT
                V_Q.enex = EntryExitLabel.ENTRY
            else:
                I_P.enex = EntryExitLabel.ENTRY
                V_P.enex = EntryExitLabel.EXIT
                I_Q.enex = EntryExitLabel.ENTRY
                V_Q.enex = EntryExitLabel.EXIT

            I_P.label = IntersectionLabel.CROSSING
            V_P.label = IntersectionLabel.CROSSING
            I_Q.label = IntersectionLabel.CROSSING
            V_Q.label = IntersectionLabel.CROSSING
    if VERBOSE:
        print(f"... {count[0]} bouncing vertex pairs split")
    
    # 6) handle CROSSING vertex candidates
    for I_P in crossing[0]:
        I_Q = I_P.neighbour
        if (I_Q in crossing[1]):
            I_P.label = IntersectionLabel.CROSSING
            I_Q.label = IntersectionLabel.CROSSING

def createResult(PP, RR, UNION = False, VERBOSE = False):
    if VERBOSE:
        print("Creating result...")
    for P in PP:
        for I in P.vertices(IteratorType.CROSSING_INTERSECTION):
            R = Polygon()
            V = I
            V.intersection = False
            while True:
                status = V.enex
                status = toggle(status)
                while True:
                    if (status is EntryExitLabel.EXIT) != UNION:
                        V = V.next
                    else:
                        V = V.prev
                    V.intersection = False

                    R.newVertex(V.node)
                    if not ((V.enex is not status) and (V is not I)):
                        break
                if V is not I:
                    V = V.neighbour
                    V.intersection = False
                if V is I:
                    break
            RR.append(R)

def cleanUpResult(RR, EPSILON, VERBOSE = False):
    if VERBOSE:
        print("Post-processing...")
    count = 0
    for R in RR:
        while (R.root is not None) and (abs(A(R.root.prev.node, R.root.node, R.root.next.node)) < EPSILON):
            R.removeVertex(R.root)
            count += 1
        if R.root is not None:
            for V in R.vertices(IteratorType.ALL):
                if abs(A(V.prev.node, V.node, V.next.node)) < EPSILON:
                    R.removeVertex(V)
                    count += 1
    if VERBOSE:
        print(f"... {count} vertices removed")

def cleanUpInput(PP, VERBOSE = False):
    if VERBOSE:
        print("Post-processing...")
    for P in PP:
        for V in P.vertices(IteratorType.ALL):
            if not V.source:
                P.removeVertex(V)


def intersect_polygons(PP, QQ, EPSILON, UNION = False, VERBOSE = False):
    RR = []
    # phase 1
    computeIntersections(PP, QQ, EPSILON,  VERBOSE) 

    # phase 2
    labelIntersections(PP, QQ, RR, UNION, VERBOSE)

    # phase 3
    createResult(PP, RR, UNION, VERBOSE)
  
    # post-processing
    cleanUpResult(RR, EPSILON, VERBOSE)
    cleanUpInput(PP, VERBOSE)
    cleanUpInput(QQ, VERBOSE)
    return RR


def labelIntersectionsLine(PP, QQ, RR, VERBOSE = False):
    if VERBOSE:
        print("Labelling intersections...")

    # 1) initial classification
    count = [0,0]
    for P in PP:
        for I in P.vertices(IteratorType.INTERSECTION):
            if I.prev is None:
                I.label = IntersectionLabel.LEFT_ON
                continue
            if I.next is None:
                I.label = IntersectionLabel.ON_LEFT
                continue
            Q_m = I.neighbour.prev
            Q_p = I.neighbour.next
            P_m = I.prev
            P_p = I.next
            Q_m_type = oracle(Q_m, P_m, I, P_p)
            Q_p_type = oracle(Q_p, P_m, I, P_p)
            if (Q_m_type is RelativePositionType.LEFT and Q_p_type is RelativePositionType.RIGHT) \
                or (Q_m_type is RelativePositionType.RIGHT and Q_p_type is RelativePositionType.LEFT):
                I.label = IntersectionLabel.CROSSING
                count[0] += 1
            if (Q_m_type is RelativePositionType.LEFT and Q_p_type is RelativePositionType.LEFT) \
                or (Q_m_type is RelativePositionType.RIGHT and Q_p_type is RelativePositionType.RIGHT):
                I.label = IntersectionLabel.BOUNCING
                count[1] += 1
            
            if (Q_p_type is RelativePositionType.IS_P_p and Q_m_type is RelativePositionType.RIGHT) \
                or (Q_m_type is RelativePositionType.IS_P_p and Q_p_type is RelativePositionType.RIGHT):
                I.label = IntersectionLabel.LEFT_ON
            if (Q_p_type is RelativePositionType.IS_P_p and Q_m_type is RelativePositionType.LEFT) \
                or (Q_m_type is RelativePositionType.IS_P_p and Q_p_type is RelativePositionType.LEFT):
                I.label = IntersectionLabel.RIGHT_ON
            if (Q_p_type is RelativePositionType.IS_P_p and Q_m_type is RelativePositionType.IS_P_m) \
                or (Q_m_type is RelativePositionType.IS_P_p and Q_p_type is RelativePositionType.IS_P_m):
                I.label = IntersectionLabel.ON_ON
            if (Q_m_type is RelativePositionType.IS_P_m and Q_p_type is RelativePositionType.RIGHT) \
                or (Q_p_type is RelativePositionType.IS_P_m and Q_m_type is RelativePositionType.RIGHT):
                I.label = IntersectionLabel.ON_LEFT
            if (Q_m_type is RelativePositionType.IS_P_m and Q_p_type is RelativePositionType.LEFT) \
                or (Q_p_type is RelativePositionType.IS_P_m and Q_m_type is RelativePositionType.LEFT):
                I.label = IntersectionLabel.ON_RIGHT        
    if VERBOSE:
        print(f"... {count[0]} crossing and {count[1]} bouncing intersection vertices")

    # 2) classify intersection chains
    count[0] = 0
    count[1] = 0
    for P in PP:
        for I in P.vertices(IteratorType.INTERSECTION):
            if (I.label is IntersectionLabel.LEFT_ON) or (I.label is IntersectionLabel.RIGHT_ON) or (I.prev is None and I.label is IntersectionLabel.ON_ON):
                if I.label is IntersectionLabel.LEFT_ON:
                    x = RelativePositionType.LEFT
                elif I.label is IntersectionLabel.RIGHT_ON:
                    x = RelativePositionType.RIGHT
                else:
                    x = RelativePositionType.RIGHT
                X = I
                I.label = IntersectionLabel.NONE
                I = I.next
                while I.label is IntersectionLabel.ON_ON:
                    I.label = IntersectionLabel.NONE
                    if I.next is None:
                        break
                    I = I.next
                if I.label is IntersectionLabel.ON_LEFT:
                    y = RelativePositionType.LEFT
                elif I.label is IntersectionLabel.ON_RIGHT:
                    y = RelativePositionType.RIGHT
                else:
                    y = x
                if (x is y) or (P.root is X) or (I.next is None):
                    chainType = IntersectionLabel.DELAYED_BOUNCING
                    count[1] += 1
                else:
                    chainType = IntersectionLabel.DELAYED_CROSSING
                    count[0] += 1
                
                X.label = chainType
                I.label = chainType
    if VERBOSE:
        print(f"... {count[0]} delayed crossings and {count[1]} delayed bouncings")

    # 3) copy labels from P to Q
    for P in PP:
        for I in P.vertices(IteratorType.INTERSECTION):
            I.neighbour.label = I.label  
            
    # 3.5) check for special cases
    noIntersection = [[], []]
    identical = [[], []]
    count[0] = 0
    count[1] = 0
    for i in range(1):
        if i == 0:
            P_or_Q = PP
            Q_or_P = QQ
        elif i == 1:
            P_or_Q = QQ
            Q_or_P = PP
        for P in P_or_Q:
            if P.noCrossingVertex(True):
                noIntersection[i].append(P)
                if P.allOnOn():
                    identical[i].append(P)
                else:
                    isInside = False
                    p = P.getNonIntersectionPoint()
                    for Q in Q_or_P:
                        if Q.pointInPoly(p):
                            isInside = not isInside
                    if isInside:
                        RR.append(P)
                        count[0] += 1
    
    for P in identical[0]:
        P_isHole = False
        for P_ in PP:
            if (P_.root is not P.root) and P_.pointInPoly(P.root.node):
                P_isHole = not P_isHole
        for Q in identical[1]:
            breakFlag = False
            for V in Q.vertices(IteratorType.ALL):
                if V is P.root.neighbour:
                    Q_isHole = False
                    for Q_ in QQ:
                        if (Q_.root is not Q.root) and Q_.pointInPoly(Q.root.node):
                            Q_isHole = not Q_isHole
                    if P_isHole != Q_isHole:
                        RR.append(P.operatorStar())
                        count[i] += 1
                    breakFlag = True 
                    break
            if breakFlag:
                break
    if VERBOSE:
        print(f"... {count[0]} interior and {count[1]} identical components added to result")

    # 4) set entry/exit flags

    for i in range(1):
        if i == 0:
            P_or_Q = PP
            Q_or_P = QQ
        elif i == 1:
            P_or_Q = QQ
            Q_or_P = PP
        for P in P_or_Q:
            if (P in noIntersection[i]):
                continue
            V = P.root
            status = EntryExitLabel.ENTRY
            for Q in Q_or_P:
                if Q.pointInPoly(V.node):
                    status = toggle(status)
            if status is EntryExitLabel.EXIT or V.intersection:
                V.enex = EntryExitLabel.ENTRY
            if V.intersection:
                status = EntryExitLabel.ENTRY
            first_chain_vertex = True
            for I in P.vertices(IteratorType.INTERSECTION, V):
                if I.label is IntersectionLabel.CROSSING:
                    I.enex = status
                    status = toggle(status)
                if I.label is IntersectionLabel.DELAYED_CROSSING:
                    if first_chain_vertex:
                        if (status is EntryExitLabel.ENTRY):
                            I.label = IntersectionLabel.CROSSING
                            I.enex = status
                        first_chain_vertex = False
                    else:
                        if (status is EntryExitLabel.EXIT):
                            I.enex = status
                            I.label = IntersectionLabel.CROSSING
                        first_chain_vertex = True
                        status = toggle(status)
                if I.label is IntersectionLabel.DELAYED_BOUNCING:            
                    if first_chain_vertex:
                        if (status is EntryExitLabel.ENTRY):
                            I.enex = status
                        first_chain_vertex = False
                    else:
                        if (status is EntryExitLabel.EXIT):
                            I.enex = status
                        first_chain_vertex = True
                    status = toggle(status)
    
def createResultLine(PP, QQ, RR, VERBOSE = False):
    if VERBOSE:
        print("Creating result...")
    for P in PP:
        R = None
        for I in P.vertices(IteratorType.ALL):
            if I.enex is EntryExitLabel.ENTRY:
                R = Line()
            if R is not None:
                R.newVertex(I.node)
            if I.enex is EntryExitLabel.EXIT:
                RR.append(R)
                R = None
        if R is not None:
            RR.append(R)

def createResultLine2(PP, QQ, RR, VERBOSE = False):
    if VERBOSE:
        print("Creating result...")
    for P in PP:
        for Q in QQ:
            R = Line()
            for I in P.vertices(IteratorType.ALL):
                if I.intersection or Q.pointInPoly(I.node):
                    R.newVertex(I.node)
                elif R.root is not None:
                    RR.append(R)
                    R = Line()
            if R.root is not None:
                RR.append(R)
     

def intersect_lines_polygons(PP, QQ, EPSILON, VERBOSE = False):
    RR = []
    # phase 1
    computeIntersections(PP, QQ, EPSILON,  VERBOSE)

    #handle last vertex
    for P in PP:
        for Q in QQ:
            for edgeQ in Q.edges(IteratorType.SOURCE):
                P1 = P.end
                Q2 = edgeQ.two
                Q1 = edgeQ.one
                if abs(A(P1.node, Q1.node, Q2.node)) < EPSILON:
                    dQ = Q2.node - Q1.node
                    PQ = Q1.node - P1.node
                    beta = -dot(PQ, dQ) / dot(dQ, dQ)
                    if (beta > EPSILON) and (beta < 1.0-EPSILON):
                        I_Q = Vertex(P1.node, beta)
                        insertVertex(I_Q, edgeQ)
                        link(P1, I_Q)
                        if VERBOSE:
                            print("+1 T")
                    elif abs(beta) <= EPSILON:
                        beta_is_0 = True
                        link(P1, Q1)
                        if VERBOSE:
                            print("+1 V")

    labelIntersectionsLine(PP, QQ, RR, VERBOSE)

    # phase 3
    createResultLine(PP, QQ, RR, VERBOSE)
  
    # post-processing
    cleanUpInput(PP, VERBOSE)
    cleanUpInput(QQ, VERBOSE)
    return RR





def main():
    P = Polygon([Node(0,0), Node(1,0), Node(1,1), Node(0,1)])
    Q1 = Polygon([Node(-0.1,0), Node(0.5,0), Node(1,1), Node(0,0.5)])#
    Q2 = Polygon([Node(-0.1,0.1), Node(0.5,0), Node(1,1), Node(0,0.5)])
    Q3 = Polygon([Node(0,0), Node(0.5,0), Node(1,1), Node(0,0.5)])
    Q4 = Polygon([Node(-0.2,0.5), Node(0.5,-0.2), Node(1.2,0.5), Node(0.5,1.2)])
    Q5 = Polygon([Node(-0.50,0.2), Node(0.5, 0.2), Node(0,0.5), Node(0,0.2)])
    #print(list(P.vertices(IteratorType.ALL)))
    #print(list(P.edges(IteratorType.ALL)))
    #print(intersect_polygons([P], [Q1], 1e-10, UNION=False, VERBOSE=True))
    L = [    Node(0.2,1), Node(0.7,1), Node(0.7,1.5), Node(0.8, 0.5), Node(1,0.5), Node(1,0), Node(2, 0)]
    L2 = L[::-1]
    LP = Line(L)
    print(intersect_lines_polygons([LP], [P], 1e-10, VERBOSE=True))
    LM = Line(L2)
    print(intersect_lines_polygons([LM], [P], 1e-10, VERBOSE=False))
    l = Line([Node(0.75, -0.5), Node(0.75, 1.25), Node(0.25, 0.75), Node(0.25, 0.25)])
    P0 = Node(0, 0)
    P1 = Node(1, 0)
    P2 = Node(0.5, 0.5)
    P3 = Node(1, 1)
    P4 = Node(0, 1)
    poly = Polygon([P0, P1, P2, P3, P4])
    #print(intersect_lines_polygons([l], [poly], 1e-10, VERBOSE=True))
if __name__ == '__main__':
    main()
