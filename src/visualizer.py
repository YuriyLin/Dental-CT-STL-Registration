"""
Visualization Module
Open3D-based visualization utilities for registration results.
"""

import numpy as np
import open3d as o3d
from typing import Optional


def visualize_registration(
    ct_pcd: o3d.geometry.PointCloud,
    stl_pcd: o3d.geometry.PointCloud,
    window_name: str = "Registration Result"
) -> None:
    """
    Visualize registration result with color-coded point clouds.
    
    Args:
        ct_pcd: CT point cloud (colored red)
        stl_pcd: Aligned STL point cloud (colored green)
        window_name: Window title
    """
    # Create copies to avoid modifying originals
    ct_vis = o3d.geometry.PointCloud(ct_pcd)
    stl_vis = o3d.geometry.PointCloud(stl_pcd)
    
    # High-contrast color scheme for clinical review
    ct_vis.paint_uniform_color([1.0, 0.4, 0.4])    # Red
    stl_vis.paint_uniform_color([0.4, 1.0, 0.4])   # Green
    
    # Visualize
    o3d.visualization.draw_geometries(
        [ct_vis, stl_vis],
        window_name=window_name,
        width=1280,
        height=720,
        point_show_normal=False
    )


def visualize_comparison(
    ct_pcd: o3d.geometry.PointCloud,
    stl_before: o3d.geometry.PointCloud,
    stl_after: o3d.geometry.PointCloud
) -> None:
    """
    Visualize before/after comparison of registration.
    
    Shows two windows:
    1. Before: CT (red) + STL original (blue)
    2. After: CT (red) + STL aligned (green)
    
    Args:
        ct_pcd: CT point cloud
        stl_before: STL point cloud before registration
        stl_after: STL point cloud after registration
    """
    print("\n=== Visualization ===")
    print("Press Q to close each window...")
    
    # Before alignment
    ct_vis1 = o3d.geometry.PointCloud(ct_pcd)
    stl_vis1 = o3d.geometry.PointCloud(stl_before)
    
    ct_vis1.paint_uniform_color([1.0, 0.4, 0.4])   # Red
    stl_vis1.paint_uniform_color([0.4, 0.4, 1.0])  # Blue
    
    print("\nShowing: BEFORE alignment (CT=Red, STL=Blue)")
    o3d.visualization.draw_geometries(
        [ct_vis1, stl_vis1],
        window_name="Before Registration",
        width=1280,
        height=720,
        point_show_normal=False
    )
    
    # After alignment
    ct_vis2 = o3d.geometry.PointCloud(ct_pcd)
    stl_vis2 = o3d.geometry.PointCloud(stl_after)
    
    ct_vis2.paint_uniform_color([1.0, 0.4, 0.4])   # Red
    stl_vis2.paint_uniform_color([0.4, 1.0, 0.4])  # Green
    
    print("\nShowing: AFTER alignment (CT=Red, STL=Green)")
    o3d.visualization.draw_geometries(
        [ct_vis2, stl_vis2],
        window_name="After Registration",
        width=1280,
        height=720,
        point_show_normal=False
    )


def save_screenshot(
    geometries: list,
    output_path: str,
    window_name: str = "Screenshot",
    width: int = 1920,
    height: int = 1080
) -> None:
    """
    Save visualization as a screenshot.
    
    Args:
        geometries: List of Open3D geometries to visualize
        output_path: Output image path (e.g., 'result.png')
        window_name: Window title
        width: Image width in pixels
        height: Image height in pixels
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height, visible=False)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    vis.poll_events()
    vis.update_renderer()
    
    vis.capture_screen_image(output_path)
    vis.destroy_window()
    
    print(f"  Screenshot saved to {output_path}")
