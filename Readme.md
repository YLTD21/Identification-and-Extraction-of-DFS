# Identification-and-Extraction-of-DFS

This project involves the use of the Seg-ASPP network for semantic segmentation predictions on large images and employs an overlap voting method to improve prediction accuracy.

## Prerequisites

Make sure you have installed the following dependencies:

- Python 3.6
- NumPy
- PyTorch
- OpenCV
- Matplotlib
- GDAL
- PIL
- torchvision

You can install these dependencies using the following command:

```bash
pip install numpy torch opencv-python matplotlib gdal pillow torchvision
```
This repository contains Python scripts for extracting river width and length parameters from binary mask images. This tool consists of three main components, each focusing on different aspects of river analysis.

## 1. Length and Curvature Extraction

The first component calculates the length and curvature of the river from a binary mask image. It uses a series of image processing techniques to preprocess the mask and extract the river's centerline. The length and curvature of these centerlines provide essential information about the river's shape.

### Usage

1. Prepare your binary mask image, ensuring it is in one of the following formats: `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, or `.tiff`.
2. Modify the following parameters in the script to meet your requirements:
   - `image_name`: Path to the input large image file.
   - `images_path`: Path to the Seg-ASPP model.
   - `model_path`: Path to the pre-trained Seg-ASPP model weights.
   - `row` and `col`: Size of each image chunk.
3. Run the script. This will process the large image in chunks and make semantic segmentation predictions using the Seg-ASPP network.
4. The predicted results will be saved as a `.tif` file.

### Notes

- Ensure that the paths to model files and large image files are correctly set.
- Depending on your requirements, you can modify other parameters such as `out_threshold` and `scale_factor`.



## 2. Centerline Extraction

The second component focuses on extracting the river's centerline and calculating its length and curvature. The centerline is often the skeleton of the river, and its length and curvature are important metrics for assessing the river's shape.

### Usage

1. Prepare a binary mask image.
2. Set the pixel length corresponding to the actual length (`pixel_length`) and provide the path to your mask image (`mask_image`).
3. Run the script to extract the centerline and calculate its length and curvature.

### Notes

- Ensure that you provide the correct pixel length and image path.
- The script extracts the centerline by performing image processing on the mask.



## 3. Width Parameter Extraction

The third component calculates the width of the river at different points along the centerline and provides various width-related measurements. These width parameters can help analyze variations and features of the river.

### Usage

1. Prepare a binary mask image.
2. Set the pixel length corresponding to the actual length (`pixlength`) and provide the path to your mask image (`image_path`).
3. Run the script to extract river width parameters.

### Notes

- Ensure that you set the pixel length and image path correctly.
- Width parameter extraction relies on the mask of the image and calculations.



---

## Integration and Results

For a comprehensive river analysis, you can integrate these components as needed and combine the results to gain a comprehensive understanding of the river's characteristics. Some parameters you can extract include:

- River length
- River curvature
- Average width
- River area

Incorporate these parameters into your research to perform in-depth analysis of river systems.

If you need more information, have questions, or suggestions, please reach out to the author.

### Author

- Author: Maolin Ye
- Email: maolinye00@gmail.com
---

## Acknowledgments

The code in this repository is based on the work of river parameter extraction and analysis. We acknowledge all contributors and organizations involved in this project.
