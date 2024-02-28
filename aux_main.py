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





def calculate_circularity(a, b, c):
    """
    Calculate the circularity of an ellipsoid based on its three principal axes.
    
    Parameters:
    - a, b, c: lengths of the ellipsoid's principal axes.
    
    Returns:
    - The circularity of the ellipsoid, using the maximum two of the three axes.
    """
    # Ensure a >= b >= c
    axes = sorted([a, b, c], reverse=True)
    
    # Use the two largest axes for circularity calculation
    a, b = axes[0], axes[1]
    
    # Calculate circularity
    circularity = (4 * a * b) / ((a + b) ** 2)
    
    return circularity


def fit_and_project_points(X):
    # Fit PCA to identify the plane
    pca = PCA(n_components=3)
    pca.fit(X)
    eig_vecs = pca.components_
    normal = eig_vecs[2, :]  # Normal vector to the plane
    centroid = np.mean(X, axis=0)
    d = -centroid.dot(normal)

    # Project points onto the plane
    projected_points = np.array([project_point_onto_plane(x, normal, centroid, d) for x in X])

    # Plotting projected points and the plane
    plot_points_and_plane(X, projected_points, centroid, normal, d)

    return projected_points

def project_point_onto_plane(point, normal, centroid, d):
    # Calculate the projection of the point onto the plane
    point_to_centroid = point - centroid
    distance = point_to_centroid.dot(normal) / np.linalg.norm(normal)
    projected_point = point - distance * normal
    return projected_point

def plot_points_and_plane(X, projected_points, centroid, normal, d):
    # Create a meshgrid for the plane visualization
    xx, yy = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 10),
                         np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10))
    zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='yellow')  # semi-transparent plane
    ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], c='b', label='Projected Points')
    ax.legend()

    plt.show()




def generate_ellipsoid_points(N, a=1, b=1, c=1):
    theta = np.random.uniform(-np.pi / 2, np.pi / 2, N)
    phi = np.random.uniform(0, 2 * np.pi, N)
    
    x = a * np.cos(theta) * np.cos(phi)
    y = b * np.cos(theta) * np.sin(phi)
    z = c * np.sin(theta)
    
    return x, y, z


def calculate_sphericity(a, b, c, p=1.6075):
    # Volume of the ellipsoid
    V = 4/3 * np.pi * a * b * c
    # Surface area approximation of the ellipsoid
    A = 4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3)**(1/p)
    # Radius of the sphere with the same volume as the ellipsoid
    r = (3*V / (4*np.pi))**(1/3)
    # Surface area of this sphere
    A_s = 4 * np.pi * r**2
    # Sphericity
    Psi = A_s / A
    return Psi



def rotate_points(points, theta, axis='z'):
    if axis == 'z':
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
    elif axis == 'y':
        R = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    return np.dot(points, R.T)

def generate_and_rotate_ellipsoid_points(N, a=1, b=1, c=1, theta=0, axis='z'):
    theta_rad = np.random.uniform(-np.pi / 2, np.pi / 2, N)
    phi_rad = np.random.uniform(0, 2 * np.pi, N)
    
    x = a * np.cos(theta_rad) * np.cos(phi_rad)
    y = b * np.cos(theta_rad) * np.sin(phi_rad)
    z = c * np.sin(theta_rad)
    
    points = np.vstack((x, y, z)).T
    rotated_points = rotate_points(points, theta, axis)
    
    return rotated_points



def ls_ellipsoid(xx, yy, zz):
    x = xx[:, np.newaxis]
    y = yy[:, np.newaxis]
    z = zz[:, np.newaxis]

    J = np.hstack((x*x, y*y, z*z, x*y, x*z, y*z, x, y, z))
    K = np.ones_like(x)

    JT = J.transpose()
    JTJ = np.dot(JT, J)
    InvJTJ = np.linalg.inv(JTJ)
    ABC = np.dot(InvJTJ, np.dot(JT, K))
    eansa = np.append(ABC, -1)

    return eansa

def polyToParams3D(vec, printMe=True):
    Amat = np.array([
        [vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0],
        [vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0],
        [vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0],
        [vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]    ]
    ])

    A3 = Amat[0:3, 0:3]
    A3inv = inv(A3)
    ofs = vec[6:9] / 2.0
    center = -np.dot(A3inv, ofs)

    Tofs = np.eye(4)
    Tofs[3, 0:3] = center
    R = np.dot(Tofs, np.dot(Amat, Tofs.T))

    R3 = R[0:3, 0:3]
    s1 = -R[3, 3]
    R3S = R3 / s1
    el, ec = eig(R3S)

    recip = 1.0 / np.abs(el)
    axes = np.sqrt(recip)

    inve = inv(ec)  # inverse is actually the transpose here

    return center, axes, inve

def printAns3D(center, axes, R, xin, yin, zin, verbose=True):
    print("\nCenter at  %10.4f,%10.4f,%10.4f" % (center[0], center[1], center[2]))
    print("Axes gains %10.4f,%10.4f,%10.4f " % (axes[0], axes[1], axes[2]))
    print("Rotation Matrix\n%10.5f,%10.5f,%10.5f\n%10.5f,%10.5f,%10.5f\n%10.5f,%10.5f,%10.5f" % (
          R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2]))

    # Transformation Matrix M omitted for brevity




def ls_ellipse(xx, yy):
    x = xx[:, np.newaxis]
    y = yy[:, np.newaxis]
    J = np.hstack((x*x, x*y, y*y, x, y))
    K = np.ones_like(x)  # Column of ones
    JT = J.transpose()
    JTJ = np.dot(JT, J)
    InvJTJ = np.linalg.inv(JTJ)
    ABC = np.dot(InvJTJ, np.dot(JT, K))
    eansa = np.append(ABC, -1)  # Append -1 to make it Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    return eansa

def polyToParams(v, printMe=True):
    Amat = np.array([
        [v[0],     v[1]/2.0, v[3]/2.0],
        [v[1]/2.0, v[2],     v[4]/2.0],
        [v[3]/2.0, v[4]/2.0, v[5]    ]
    ])
    A2 = Amat[0:2, 0:2]
    A2Inv = inv(A2)
    ofs = v[3:5] / 2.0
    cc = -np.dot(A2Inv, ofs)
    Tofs = np.eye(3)
    Tofs[2, 0:2] = cc
    R = np.dot(Tofs, np.dot(Amat, Tofs.T))
    R2 = R[0:2, 0:2]
    s1 = -R[2, 2]
    RS = R2 / s1
    el, ec = eig(RS)
    recip = 1.0 / np.abs(el)
    axes = np.sqrt(recip)
    rads = np.arctan2(ec[1, 0], ec[0, 0])
    deg = np.degrees(rads)
    return (cc[0], cc[1], axes[0], axes[1], deg, ec)

def printAns(pv, xin, yin, verbose):
    if verbose:
        print('\nPolynomial coefficients, F term is -1:\n', pv)
        nrm = np.sqrt(np.dot(pv, pv))
        enrm = pv / nrm
        if enrm[0] < 0.0:
            enrm = -enrm
        print('\nNormalized Polynomial Coefficients:\n', enrm)
    params = polyToParams(pv, verbose)
    print("\nCenter at  %10.4f,%10.4f" % (params[0], params[1]))
    print("Axes lengths %10.4f,%10.4f" % (params[2], params[3]))
    print("Tilt Degrees %10.4f" % (params[4]))


def calculate_axis_skewness(axis_coordinates):
    """
    Calculate the skewness of the distribution of vertex coordinates along a principal axis.
    
    Parameters:
    - axis_coordinates: numpy array of vertex coordinates projected onto one of the principal axes.
    
    Returns:
    - Skewness value for the axis.
    """
    n = len(axis_coordinates)
    mean = np.mean(axis_coordinates)
    std_dev = np.std(axis_coordinates)
    skewness = (np.sum((axis_coordinates - mean) ** 3) / n) / (std_dev ** 3) if std_dev > 0 else 0
    return skewness

def geometric_skewness(vertices, center):
    """
    Computes a measure of geometric skewness for a mesh.

    Parameters:
    - vertices: numpy array of shape (n, 3), representing the mesh vertices.
    - center: The (x, y, z) center of the fitted ellipsoid.

    Returns:
    - A float representing the geometric skewness of the mesh.
    """
    # Compute distances from the center to each vertex
    distances = np.sqrt(np.sum((vertices - center) ** 2, axis=1))
    
    # Compute variance of these distances
    variance = np.var(distances)
    
    return variance


