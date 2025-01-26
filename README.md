# Panorama Stitching From Scratch

This project implements a panorama stitching algorithm in Python without relying on external libraries for feature detection, matching, or image alignment. The implementation includes manual corresponding points selection, homography computation, image warping, and blending to create a seamless panorama.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)

## Overview

This project was created as part of **Computer Vision Homework 2** to demonstrate the understanding and implementation of fundamental image stitching techniques. The pipeline includes:
- Manual selection of corresponding points.
- Homography computation using Singular Value Decomposition (SVD).
- Warping and blending images to produce a final panorama.

## Features

- **Manual Corresponding Points Selection**: Users can manually select matching points between two images.
- **Homography Computation**: Computes the transformation matrix using SVD to align the images.
- **Image Warping**: Maps pixels from one image to the other using the homography matrix.
- **Blending**: Combines the aligned images into a seamless panorama.
- **Multiple Stitching Methods**:
  - Left-to-right stitching.
  - Middle-out stitching.
  - First-out-then-middle stitching.

## How It Works

1. **Select Corresponding Points**:  
   Users select matching points between images manually, ensuring accurate alignment.

2. **Compute Homography**:  
   The homography matrix is computed to transform one image into the coordinate system of the other.

3. **Warp Images**:  
   Images are warped using the homography matrix to align them correctly.

4. **Blend Images**:  
   The overlapping areas of the images are blended to form a seamless panorama.

5. **Stitching Methods**:  
   Different strategies for stitching multiple images are provided to handle different datasets effectively.

## Requirements

This project requires the following Python libraries:
- `numpy`
- `matplotlib`
- `opencv-python`
- `scipy`

Install them using:
```bash
pip install numpy matplotlib opencv-python scipy
