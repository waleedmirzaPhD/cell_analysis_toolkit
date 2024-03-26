import math
import os
import re
import numpy as np
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize
from sklearn.decomposition import PCA



def find_farthest_points(points):
    """
    Find the two points in the set that are farthest apart.

    This function iterates over all pairs of points in the given set, calculates the Euclidean distance between each pair,
    and identifies the pair with the maximum distance. This approach ensures that the two points defining the greatest
    span within the set are selected, regardless of the points' arrangement or distribution.

    Parameters:
    - points (array-like): A collection of points (e.g., as NumPy arrays, lists, or tuples) where each point is defined
      by its coordinates in space (e.g., (x, y) for 2D, (x, y, z) for 3D).

    Returns:
    - tuple: A pair of points (each point represented by its coordinates) that are farthest apart from each other in the set.
      Returns `None` if the points list is empty or contains a single point.

    Approach:
    - Initialize `max_distance` with 0 to track the maximum found distance.
    - Initialize `farthest_points` with `None` to hold the pair of points that are farthest apart.
    - Use a nested loop to iterate over all unique pairs of points: for each point `point1`, compare it with each subsequent
      point `point2`.
    - Calculate the Euclidean distance between `point1` and `point2` using `np.linalg.norm(point1 - point2)`.
    - If the calculated distance is greater than `max_distance`, update `max_distance` and `farthest_points` with the current
      pair and distance.
    - After examining all pairs, return `farthest_points` as the result.
    """
    max_distance = 0  # Initialize the variable to track the maximum distance found.
    farthest_points = None  # Initialize the variable to hold the farthest points.
    for i, point1 in enumerate(points):  # Iterate over each point to compare it with others.
        for point2 in points[i+1:]:  # Iterate over the remaining points for comparison.
            distance = np.linalg.norm(point1 - point2)  # Calculate the Euclidean distance between the two points.
            if distance > max_distance:  # Check if this distance is the largest found so far.
                max_distance = distance  # Update the maximum distance.
                farthest_points = (point1, point2)  # Update the farthest points with the current pair.
    return farthest_points  # Return the pair of points that are farthest apart.



def perpendicular_distance(point, line):
    """
    Calculate the perpendicular distance from a point to a line.
    
    This function computes the perpendicular (shortest) distance from a given point to a specified line. 
    The line is defined by two points (point1 and point2), and the distance is calculated to another point.
    
    Parameters:
    - point (np.array): A numpy array representing the coordinates of the point.
    - line (tuple): A tuple containing two numpy arrays representing the coordinates of the two points 
      that define the line.
      
    Returns:
    - float: The perpendicular distance from the point to the line.
    
    The calculation uses the cross product of vectors defined by the line and the point to the line, 
    divided by the magnitude of the vector defining the line, to determine the perpendicular distance.
    This approach leverages vector operations for efficiency and clarity.
    """

    # Extract the two points that define the line
    point1, point2 = line
    
    # Calculate the vector from point1 to point2 (defining the line)
    # and the vector from point1 to the given point
    # Then, compute the cross product of these vectors
    # The magnitude of this cross product gives the area of the parallelogram formed by the vectors
    
    # Calculate the perpendicular distance as the area of the parallelogram divided by the length of the line
    # This effectively gives the height of the parallelogram, which is the perpendicular distance sought
    return np.abs(np.cross(point2-point1, point1-point)) / np.linalg.norm(point2-point1)


