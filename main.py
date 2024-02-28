import math
import os
import re
from  aux_main import ls_ellipsoid, polyToParams3D,calculate_circularity,calculate_axis_skewness,geometric_skewness
import numpy as np
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from numpy.linalg import eig, inv



if __name__ == "__main__":
    prefix = 'mesh' # Prefix of mesh
    mesh_folder_path = 'cell'  # Adjust to the path where your STL files are stored
    file_pattern = re.compile(fr'{prefix}_(\d+).stl')

    mesh_files = sorted(
        (f for f in os.listdir(mesh_folder_path) if file_pattern.match(f)),
        key=lambda x: int(file_pattern.match(x).group(1))   
    )

    circularity_values = []

    for file_name in mesh_files:
        file_path = os.path.join(mesh_folder_path, file_name)
        your_mesh = mesh.Mesh.from_file(file_path)

        vertices = np.vstack((your_mesh.v0, your_mesh.v1, your_mesh.v2))
            # Perform PCA on the vertices
        xx, yy, zz = vertices[:, 0], vertices[:, 1], vertices[:, 2]

        coeffs = ls_ellipsoid(xx, yy, zz)
        center, axes, rotation_matrix = polyToParams3D(coeffs)

        circularity = calculate_circularity(*axes)
        circularity_values.append(circularity)


        pca = PCA(n_components=3)
        pca.fit(vertices)
        transformed_vertices = pca.transform(vertices)  # Project vertices onto principal axes

        # Calculate skewness along each principal axis
        skewness_x = calculate_axis_skewness(transformed_vertices[:, 0])
        skewness_y = calculate_axis_skewness(transformed_vertices[:, 1])
        skewness_z = calculate_axis_skewness(transformed_vertices[:, 2])
        skewness = geometric_skewness(vertices, center)


        print(f" ########  For the File {file_name}  ########")
        print(f"Circularity = {circularity}")
        print(f" Skewness in principal directions: X={skewness_x}, Y={skewness_y}, Z={skewness_z}")
        print("Overall Skewness:", skewness)



   