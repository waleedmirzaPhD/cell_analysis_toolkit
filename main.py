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
        ##By determining the principal components, the fit process prepares the PCA model to reduce the dimensionality of the dataset. The number of principal components to keep can be specified by the n_components parameter during the PCA initialization. In this case, setting n_components=2 indicates that only the first two principal components are retained, which are the ones associated with the largest eigenvalues and thus capture the most variance.
        pca = PCA(n_components=2)
        ##The fit method analyzes the vertices, which typically represent a dataset of multidimensional points (e.g., vertices of 3D objects in space). The goal is to learn the statistical properties of the dataset, specifically the directions (in feature space) that account for the most variance in the data.
        pca.fit(vertices)
        ##It calculates the mean of each feature (dimension) of the dataset. This mean is used to center the dataset by subtracting the mean of each dimension from all data points, ensuring the centered data has a mean of zero.
        mean = pca.mean_
        ##It identifies the principal components, which are the directions in which the data varies the most. This is achieved by computing the eigenvalues and eigenvectors of the covariance matrix of the centered dataset or using Singular Value Decomposition (SVD) on the centered data matrix. The eigenvectors represent the directions of the principal components, while the eigenvalues indicate the variance captured by each principal component.
        components = pca.components_
        ##Purpose: This line centers the dataset by subtracting the mean of each dimension from all data points. The mean is calculated during the pca.fit(vertices) step and represents the average of the data along each dimension. By subtracting this mean, each dimension of the data is centered around the origin (0) of the coordinate system. This is a crucial preprocessing step in PCA to ensure that the principal components reflect the directions of maximum variance correctly.
        ##Outcome: The result, points_centered, is a new dataset where the geometric center is aligned with the origin of the coordinate system. This transformation is necessary for the accurate application of PCA, facilitating the projection of the data onto the principal components in the next step.
        points_centered = vertices - mean
        ##This line projects the centered data points onto the principal components. The components matrix obtained from pca.fit(vertices) contains the principal components as its rows. Transposing this matrix (components.T) aligns the principal components as columns. The dot product between points_centered and components.T calculates the projection of each data point onto the new axes defined by the principal components.
        projected = np.dot(points_centered, components.T)

        # Step 2: Calculate the Convex Hull in 2D
        ## computes the geometric boundary encapsulating the dataset in its newly transformed space, offering a tool for further morphological analysis and insights into the data's structure.
        hull = ConvexHull(projected)
        ##Here, the area of the convex hull (in a 2D space) is obtained using hull.volume. Despite the name, volume represents the area of the hull in 2D contexts. This is a key geometric property, indicative of the spread or coverage of the dataset.
        area = hull.volume  # In 2D, hull.volume gives the area
        ##This line retrieves the perimeter (the total length of the edges that form the convex hull) using hull.area. It's worth noting that in the context of the convex hull object, hull.area refers to the perimeter in 2D spaces.
        perimeter = hull.area
        # Calculate the circularity of the convex hull, a dimensionless metric to assess the shape's similarity to a perfect circle. Circularity is defined as (4 * np.pi * area) / (perimeter ** 2). For a perfect circle, circularity equals 1. Values less than 1 indicate shapes that deviate from circular, with smaller values suggesting more elongated or irregular shapes. This metric is useful in shape analysis,to understand the geometric properties of the dataset or the object it represents.
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        #Step 3: Further Postprocessing 
        ##  The farthest points on this hull are found to determine the shape's longest dimension. 
        farthest_points = find_farthest_points(projected[hull.vertices])
        ## For each vertex on the convex hull, the perpendicular distance to the line defined by these farthest points is calculated, effectively measuring the width of the shape at various points along its length. These widths are gathered into a list, providing a detailed view of the shape's variability and allowing for further morphological analysis such as identifying the minimum, maximum, or average width, thereby offering insights into the dataset's geometric and spatial properties.
        widths = [perpendicular_distance(point, farthest_points) for point in projected[hull.vertices]]
        ## Ensure minimum width is not zero
        ## Filter out any zero values from widths; consider a small epsilon to catch near-zero values due to computational precision
        epsilon = 1e-6  # Small threshold to consider as zero
        filtered_widths = [w for w in widths if w > epsilon]
        ## Compute the maximum, minimum, and average width
        max_width = np.max(filtered_widths)
        min_width = np.min(filtered_widths)
        average_width = np.mean(filtered_widths)

        #Step 4: Writing results
        ## Add the results to the DataFrame
        results_df.loc[index] = [file_name, circularity,max_width,min_width,average_width]
        ## Plot the 2D Convex Hull for visualization
        plt.figure()
        plt.plot(projected[:, 0], projected[:, 1], 'o', markersize=3)  # Plot points
        for simplex in hull.simplices:
            plt.plot(projected[simplex, 0], projected[simplex, 1], 'k-')  # Plot hull edges
        plt.title(f"Convex Hull for {file_name}")
        plt.xlabel('Principle Component 1')
        plt.ylabel('Principle Component 2')
        plt.axis('equal')
        plt.savefig(os.path.join(results_dir, f"{file_name}.png"))
        plt.close()  # Close the plot to free memory
    # Write the DataFrame to an Excel file
    # Write the DataFrame to an Excel file in the 'results' directory
    results_df.to_excel(os.path.join(results_dir, 'mesh_results.xlsx'), index=False)

   