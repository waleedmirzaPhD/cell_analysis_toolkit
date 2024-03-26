
# Overview

This manual guides you through the usage of a Python script designed for analyzing the morphology of pericyte meshes. The script quantifies circularity, sphericity, maximum width, minimum width, and average width of 3D-rendered pericyte meshes from STL files. It utilizes PCA for dimension reduction, Convex Hull analysis for shape description, and custom algorithms for width measurement and shape characterization.

## Requirements

- Python 3.x
- Libraries: math, os, re, numpy, pandas, matplotlib, scipy, sklearn
- Auxiliary Python files: `aux_main.py` containing `find_farthest_points` and `perpendicular_distance` functions.
## Setup

### Install Required Libraries

Ensure all required Python libraries are installed. Use the command:
`pip install numpy pandas matplotlib scipy sklearn python-stl`
### Prepare the `aux_main.py` File

Ensure `aux_main.py` is in the same directory as the script, containing definitions for `find_farthest_points` and `perpendicular_distance`.

## Usage Instructions

### Prepare STL Files

Place all your STL files in a designated folder. The files should be named with a common prefix followed by an underscore and a numeric identifier (e.g., `Pericyte_material_1.stl`).

### Configure the Script

## Troubleshooting

- **Library Installation Issues:** Ensure you have the correct versions of Python and pip. Try updating pip using `pip install --upgrade pip` before installing the required libraries.
- **File Naming Convention:** Ensure STL files follow the naming convention specified in the setup. Incorrect file names may lead to files being ignored.
- **Script Errors:** Check that all auxiliary files (e.g., `aux_main.py`) are correctly placed and that all necessary functions are defined. Ensure no syntax errors in the script.

## Advanced Configuration

- **Custom Analysis:** Modify the script to include additional morphological parameters or different analysis techniques as needed.
- **Visualization Enhancements:** Adjust the matplotlib code to enhance the visualization of the Convex Hull or to add additional plot features.

## Support

For further assistance or to report bugs, please contact the script's author or the support team provided with the software documentation. You can reach out via email at `waleed.mirza@embl.es`.


- **Prefix:** Update the `prefix` variable to match the common prefix of your STL files.
- **Mesh Folder Path:** Change the `mesh_folder_path` variable to the path of the folder containing your STL files.

### Running the Script

1. Open a terminal or command prompt.
2. Navigate to the directory containing the script.
3. Run the script using Python:
`python3 main.py`
4. The script will process each STL file, perform the analysis, and save the results.

### Results

- **Plots:** For each STL file, a 2D plot of the Convex Hull projection will be saved in the results directory.
- **Excel File:** An Excel file named `mesh_results.xlsx` containing the calculated parameters for each mesh will be generated in the results directory.





