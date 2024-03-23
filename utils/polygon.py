from operator import is_
from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Rect, Vector
from supervision.tools.detections import Detections
import cv2
import numpy as np

class Polygon:
    def __init__(self, points):
        self.points = points
        # edges = list of vectors, 1-2 2-3 3-4 ... (n-1)-n n-1
        self.edges = [Vector(start=points[i], end=points[i+1]) for i in range(len(points)-1)] + [Vector(start=points[-1], end=points[0])]

    def is_in(self, point: Point) -> bool:
        """
        Check if a point is inside the polygon.

        :param point: Point : The point to check.
        :return: bool : True if the point is inside the polygon, False otherwise.
        """
        # see how many edges the vector from the point to the right crosses
        # if the number is odd, the point is inside the polygon
        # if the number is even, the point is outside the polygon
        # if the number is 0, the point is on the edge of the polygon
        counter = 0
        for edge in self.edges:
            if point.y > min(edge.start.y, edge.end.y):
                if point.y <= max(edge.start.y, edge.end.y):
                    if point.x <= max(edge.start.x, edge.end.x):
                        if edge.start.y != edge.end.y:
                            x_intersection = (point.y - edge.start.y) * (edge.end.x - edge.start.x) / (edge.end.y - edge.start.y) + edge.start.x
                            if edge.start.x == edge.end.x or point.x <= x_intersection:
                                counter += 1
        return counter % 2 != 0
    
    