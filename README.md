# 3D Power Line and Pylon Extraction from LiDAR Data

This project is a Python implementation of the research paper: **"An Automatic Pylon and Power Line Extraction Method for Power Line Inspection Using Airborne LiDAR Data"** (Remote Sensing, 2019). It provides a pipeline to automatically detect and extract 3D models of transmission pylons and power lines from airborne LiDAR point cloud data.

The original paper can be found in the project root (`remotesensing-11-02600-v4.pdf`).

## Dependencies

The project is written in Python 3 and requires the following libraries. It is highly recommended to use a virtual environment.

*   `numpy`
*   `scipy`
*   `laspy`
*   `matplotlib`
*   `scikit-learn`
*   `numba`

You can install these dependencies using pip:
```bash
pip install numpy scipy laspy matplotlib scikit-learn numba
```
It would be best to create a `requirements.txt` file for easier dependency management.

## Usage

The main entry point for the extraction pipeline is `src/main.py`. You can run it from the command line, specifying the input `.las` file and an output directory.

```bash
# Example usage
python src/main.py --input /path/to/your/data.las --output /path/to/output_dir
```
*Note: The command-line arguments `--input` and `--output` are assumed based on the code structure. You may need to adjust the script `src/main.py` to handle command-line arguments properly.*

The script `src/test_pipeline.py` can be used for testing parts of the pipeline and visualizing intermediate results.

## Project Structure

The source code is organized into several modules within the `src/` directory:

-   `main.py`: The main script that orchestrates the entire extraction pipeline.
-   `data_structures.py`: Defines custom data structures used throughout the project.
-   `preprocessing.py`: Handles initial point cloud processing, such as ground filtering and down-sampling.
-   `pylon_extraction.py`: Contains the logic for detecting and extracting transmission pylons from the point cloud.
-   `power_line_extraction.py`: Implements the algorithms for identifying and modeling power lines, likely using catenary curve fitting.
-   `feature_calculation.py`: Calculates various geometric features from point cloud segments.
-   `optimization.py`: Contains optimization routines, possibly for model fitting.
-   `reconstruction.py`: Handles the 3D reconstruction of the final pylon and power line models.
-   `test_pipeline.py`: A script for development, testing, and visualization.

