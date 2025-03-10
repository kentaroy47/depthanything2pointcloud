# depthanything2pointcloud

This repository contains a Python tool for converting depth maps generated by [Depth Anything](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2) into 3D point clouds and visualize with Open3D.


https://github.com/user-attachments/assets/219c4d26-0b66-4368-abc0-c331994a85e5




## Overview

[Depth Anything](https://github.com/LiheYoung/Depth-Anything) is a powerful monocular depth estimation model that can generate depth maps from single images. This tool takes those depth maps and converts them into 3D point clouds, allowing for visualization with open3d.

## Usage

### Step 1: Generate a Depth Map

1. Visit the [Depth Anything V2 Hugging Face Space](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2)
2. Upload your image and generate a depth map
3. Download both the depth map and your original image

### Step 2: Run the Converter

1. Save the downloaded depth map (16-bit PNG) and your original image in the same directory as the script
2. Update the file paths in the script:

```python
depth_file = "your_depth_map.png"  # Your downloaded depth map
rgb_file = "your_original_image.png"  # Your original image
```

3. Run the script:

```bash
python depth_to_pointcloud.py
```

The tool will:
- Load your depth map and original image
- Convert the depth map to a 3D point cloud
- Display the point cloud in an interactive 3D viewer
- Save the point cloud as a PLY file for use in other 3D software

## Customization

You can adjust several parameters in the script to improve results:

```python
# Camera parameters
fx = fy = max(depth.shape) * 0.8  # Focal length approximation
cx = width / 2  # Principal point X
cy = height / 2  # Principal point Y

# Depth conversion parameters
depth_scale = 0.1  # Adjust to change the scale of the Z dimension
depth_trunc = 5.0  # Maximum depth in meters
```
