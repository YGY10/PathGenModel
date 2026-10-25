"""Small immutable 2-D primitives; World coordinates are metres."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Point:
    x: float
    y: float
    def __post_init__(self):
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError('Non-finite point')

@dataclass(frozen=True)
class Segment:
    a: Point
    b: Point
    def __post_init__(self):
        if math.dist((self.a.x, self.a.y), (self.b.x, self.b.y)) < 1e-7:
            raise ValueError('Degenerate segment')

@dataclass(frozen=True)
class Box:
    """Four ordered corner points, supports rotated/non-rectangular quads."""
    corners: tuple[Point, Point, Point, Point]
    def __post_init__(self):
        p = self.corners
        if len(p) != 4:
            raise ValueError('Box must have exactly 4 corners')
        cross = lambda a,b,c: (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x)
        z=[cross(p[i],p[(i+1)%4],p[(i+2)%4]) for i in range(4)]
        if min(abs(v) for v in z)<1e-8 or not (all(v>0 for v in z) or all(v<0 for v in z)):
            raise ValueError('Box must be non-degenerate convex quadrilateral')

# A polyline is an ordered tuple of Points; its edges are Segments.
def polyline(points):
    result=tuple(Point(float(x), float(y)) for x,y in points)
    if len(result)<2 or any(math.dist((a.x,a.y),(b.x,b.y))<1e-7 for a,b in zip(result,result[1:])):
        raise ValueError('Polyline requires at least two distinct consecutive points')
    return result
