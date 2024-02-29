import math
import os
import re
import numpy as np
from aux_main import  find_farthest_points,  perpendicular_distance
from stl import mesh
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt 

# Ensure the 'results' directory exists
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

if __name__ == "__main__":
    prefix = 'Pericyte_material'  # Prefix of mesh
    mesh_folder_path = 'cell'  # Adjust to the path where your STL files are stored
    file_pattern = re.compile(fr'{prefix}_(\d+).stl')
    mesh_files = sorted(
        (f for f in os.listdir(mesh_folder_path) if file_pattern.match(f)),
        key=lambda x: int(file_pattern.match(x).group(1))
    )

    # Initialize a DataFrame to store filenames, circularities, and sphericities
    results_df = pd.DataFrame(columns=['FileName', 'Circularity', 'Sphericity', 'MaxWidth', 'MinWidth', 'AverageWidth'])

    for index, file_name in enumerate(mesh_files):
        file_path = os.path.join(mesh_folder_path, file_name)
        your_mesh = mesh.Mesh.from_file(file_path)
        vertices = np.vstack((your_mesh.v0, your_mesh.v1, your_mesh.v2))

        # Step 1: Fit a plane using PCA and project points onto the plane
        pca = PCA(n_components=2)
        pca.fit(vertices)
        mean = pca.mean_
        components = pca.components_
        points_centered = vertices - mean
        projected = np.dot(points_centered, components.T)

        # Step 2: Calculate the Convex Hull in 2D
        hull = ConvexHull(projected)
        area = hull.volume  # In 2D, hull.volume gives the area
        perimeter = hull.area
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Step 3: Calculate Sphericity from 3D Convex Hull
        hull3D = ConvexHull(vertices)
        volume3D = hull3D.volume
        surface_area = hull3D.area
        sphericity = np.pi**(1/3) * (6 * volume3D)**(2/3) / surface_area



        # Find the two points that are farthest apart
        farthest_points = find_farthest_points(projected[hull.vertices])

        # Calculate the perpendicular distances of all hull points to the line formed by farthest points
        widths = [perpendicular_distance(point, farthest_points) for point in projected[hull.vertices]]
        # Ensure minimum width is not zero
        # Filter out any zero values from widths; consider a small epsilon to catch near-zero values due to computational precision
        epsilon = 1e-6  # Small threshold to consider as zero
        filtered_widths = [w for w in widths if w > epsilon]

        # Compute the maximum, minimum, and average width
        max_width = np.max(filtered_widths)
        min_width = np.min(filtered_widths)
        average_width = np.mean(filtered_widths)

        pca = PCA(n_components=3)
        pca.fit(vertices)
        transformed_vertices = pca.transform(vertices)  # Project vertices onto principal axes

        # Calculate skewness along each principal axis
        skewness_x = calculate_axis_skewness(transformed_vertices[:, 0])
        skewness_y = calculate_axis_skewness(transformed_vertices[:, 1])
        skewness_z = calculate_axis_skewness(transformed_vertices[:, 2])
        skewness = geometric_skewness(vertices, center)

        # Add the results to the DataFrame
        results_df.loc[index] = [file_name, circularity, sphericity,max_width,min_width,average_width]

        # Plot the 2D Convex Hull for visualization
        plt.figure()
        plt.plot(projected[:, 0], projected[:, 1], 'o', markersize=3)  # Plot points
        for simplex in hull.simplices:
            plt.plot(projected[simplex, 0], projected[simplex, 1], 'k-')  # Plot hull edges
        plt.title(f"Convex Hull for {file_name}")
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.axis('equal')
        plt.savefig(os.path.join(results_dir, f"{file_name}.png"))
        plt.close()  # Close the plot to free memory

        #print(f" ########  For the File {file_name}  ########")
        #print(f"Circularity = {circularity}, Sphericity = {sphericity}")

    # Write the DataFrame to an Excel file
    # Write the DataFrame to an Excel file in the 'results' directory
    results_df.to_excel(os.path.join(results_dir, 'mesh_results.xlsx'), index=False)

   