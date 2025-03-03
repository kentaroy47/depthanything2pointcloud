import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

def depth_to_pointcloud(depth_map, rgb_image=None, fx=1000, fy=1000, cx=None, cy=None, depth_scale=1.0, depth_trunc=5.0):
    """
    Convert a depth map (from Depth Anything or similar models) to a colored point cloud.
    
    Parameters:
    -----------
    depth_map : numpy.ndarray
        The depth map image (float values)
    rgb_image : numpy.ndarray, optional
        The corresponding RGB image for coloring the point cloud
    fx, fy : float
        Focal length in pixels in x and y directions
    cx, cy : float, optional
        Principal point offset in pixels. If None, image center is used.
    depth_scale : float
        Scale factor to convert depth map values to meters
    depth_trunc : float
        Maximum depth threshold in meters
        
    Returns:
    --------
    open3d.geometry.PointCloud
        The resulting point cloud
    """
    # Get image dimensions
    height, width = depth_map.shape
    
    # Set principal point to image center if not provided
    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2
    
    # Create an Open3D depth image
    depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
    
    # Create camera intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    
    # Convert depth image to point cloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, 
        intrinsic, 
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        project_valid_depth_only=True
    )
    
    # Add colors if RGB image is provided
    if rgb_image is not None:
        # Ensure RGB image has the same dimensions as depth map
        if rgb_image.shape[:2] != depth_map.shape:
            rgb_image = cv2.resize(rgb_image, (width, height))
        
        # Convert to RGB (from BGR if from OpenCV)
        if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Create Open3D color image
        color_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
        
        # Create RGBD image and convert to point cloud
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, 
            depth_o3d, 
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    # Flip the orientation for better visualization
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    return pcd

def process_depth_anything_output(depth_file, rgb_file=None):
    """
    Process depth output from Depth Anything model and convert to point cloud.
    
    Parameters:
    -----------
    depth_file : str
        Path to the depth map file
    rgb_file : str, optional
        Path to the corresponding RGB image
        
    Returns:
    --------
    open3d.geometry.PointCloud
        The resulting point cloud
    """
    # Load depth map as 16-bit
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    
    if depth is None:
        raise FileNotFoundError(f"Could not load depth map from {depth_file}")
    
    print(f"Loaded depth map with dtype: {depth.dtype}, min: {depth.min()}, max: {depth.max()}")
    
    # Ensure we have 16-bit data and convert to float
    if depth.dtype == np.uint16:
        # 16-bit raw depth map (0-65535)
        # Normalize to 0-1 range
        depth = depth.astype(np.float32) / 65535.0
    elif depth.dtype == np.uint8:
        # Convert 8-bit to float and scale
        print("Warning: Expected 16-bit depth map but got 8-bit")
        depth = depth.astype(np.float32) / 255.0
    
    # Invert if necessary (Depth Anything might have closer objects brighter)
    # For Depth Anything, typically closer objects are darker (smaller values)
    if np.mean(depth[depth.shape[0]//2:, :]) < np.mean(depth[:depth.shape[0]//2, :]):
        print("Inverting depth map (assuming closer objects should be darker)")
        depth = 1.0 - depth
    
    # Load RGB image if provided
    rgb = None
    if rgb_file:
        rgb = cv2.imread(rgb_file)
        if rgb is None:
            print(f"Warning: Could not load RGB image from {rgb_file}")
    
    # Convert to point cloud
    # Adjust these parameters based on your data
    fx = fy = max(depth.shape) * 0.8  # Use approximate focal length (0.8 factor for better results with Depth Anything)
    
    # For 16-bit raw depth maps, we need to adjust these values
    depth_scale = 0.1    # Scaling factor - adjust for your scene
    depth_trunc = 5.0    # Maximum depth in meters
    
    pcd = depth_to_pointcloud(
        depth, rgb, 
        fx=fx, fy=fy, 
        depth_scale=depth_scale, 
        depth_trunc=depth_trunc
    )
    
    return pcd, depth

def main():
    # File paths
    depth_file = "tmpoyxjgr27.png"  # Your Depth Anything output
    rgb_file = "baby.png"          # Corresponding RGB image
    
    print(f"Processing depth map from {depth_file}...")
    pcd, depth = process_depth_anything_output(depth_file, rgb_file)
    
    # Display statistics
    print(f"Created point cloud with {len(pcd.points)} points")
    
    # Optional: Apply statistical outlier removal for cleaner point cloud
    print("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Visualize the point cloud
    print("Visualizing point cloud...")
    
    # Use the simpler visualization method without zoom parameter:
    o3d.visualization.draw_geometries([pcd], 
                                     window_name="Depth Anything Point Cloud",
                                     width=1024, 
                                     height=768,
                                     point_show_normal=False)
    
    # Or the more configurable Visualizer:
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window("Depth Anything Point Cloud", width=1024, height=768)
    vis.add_geometry(pcd)
    
    # Improve visualization
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    # Set a better viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    
    # Update view
    vis.run()
    vis.destroy_window()
    """
    
    # Save the point cloud
    o3d.io.write_point_cloud("depth_anything_pointcloud.ply", pcd)
    print("Point cloud saved as 'depth_anything_pointcloud.ply'")
    
    # Show the depth map
    plt.figure(figsize=(10, 5))
    plt.imshow(depth, cmap='plasma')
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.show()

if __name__ == "__main__":
    main()