"""
Dental CT-STL Registration Pipeline
A stability-first, clinically-viable registration system for aligning 
intraoral STL scans to CT coordinate systems.
"""

from .dicom_loader import load_dicom_series, extract_teeth_surface, volume_to_pointcloud
from .stl_loader import load_stl, prepare_pointcloud_for_icp
from .registration import (
    load_landmarks,
    compute_rigid_transform_svd,
    apply_transform,
    refine_with_icp
)
from .visualizer import visualize_registration, visualize_comparison
from .utils import (
    compute_rmse,
    compute_inlier_ratio,
    evaluate_registration_quality,
    save_transform
)

__version__ = "1.0.0"
__all__ = [
    # DICOM processing
    "load_dicom_series",
    "extract_teeth_surface",
    "volume_to_pointcloud",
    # STL processing
    "load_stl",
    "prepare_pointcloud_for_icp",
    # Registration
    "load_landmarks",
    "compute_rigid_transform_svd",
    "apply_transform",
    "refine_with_icp",
    # Visualization
    "visualize_registration",
    "visualize_comparison",
    # Utils
    "compute_rmse",
    "compute_inlier_ratio",
    "evaluate_registration_quality",
    "save_transform",
]


def run_registration_pipeline(
    dicom_dir,
    stl_path,
    landmarks_path,
    output_dir,
    hu_threshold=1200,
    visualize=True
):
    """
    Wrapper function to run the complete registration pipeline.
    
    Args:
        dicom_dir: Path to directory containing DICOM files
        stl_path: Path to STL file
        landmarks_path: Path to landmarks JSON file
        output_dir: Output directory for results
        hu_threshold: HU threshold for tooth extraction (default: 1200)
        visualize: Whether to show visualization (default: True)
    
    Returns:
        dict: Registration results including transformation matrix and metrics
    """
    import os
    import numpy as np
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CT and extract teeth
    print("Loading CT DICOM series...")
    ct_image = load_dicom_series(dicom_dir)
    
    print(f"Extracting teeth surface (HU threshold: {hu_threshold})...")
    ct_mesh = extract_teeth_surface(ct_image, hu_threshold=hu_threshold)
    
    print("Converting CT mesh to point cloud...")
    ct_pcd = volume_to_pointcloud(ct_mesh, voxel_size=0.4)
    
    # Load STL
    print("Loading STL file...")
    stl_mesh = load_stl(stl_path)
    
    print("Preparing STL point cloud...")
    stl_pcd = prepare_pointcloud_for_icp(stl_mesh, target_points=8000)
    
    # Load landmarks and perform registration
    print("Loading landmarks...")
    ct_landmarks, stl_landmarks = load_landmarks(landmarks_path)
    
    print("Computing rigid transformation (SVD)...")
    initial_transform = compute_rigid_transform_svd(stl_landmarks, ct_landmarks)
    
    print("Applying initial transformation...")
    stl_pcd_aligned = apply_transform(stl_pcd, initial_transform)
    
    print("Refining with ICP (two-stage: 2.0mm → 1.0mm)...")
    final_transform, icp_result = refine_with_icp(
        stl_pcd_aligned, ct_pcd, initial_transform
    )
    
    # Apply final transform
    stl_pcd_final = apply_transform(stl_pcd, final_transform)
    
    # Evaluate quality
    print("Evaluating registration quality...")
    rmse = compute_rmse(stl_pcd_final, ct_pcd)
    inlier_ratio = compute_inlier_ratio(stl_pcd_final, ct_pcd, threshold=1.0)
    quality = evaluate_registration_quality(rmse, inlier_ratio)
    
    print(f"\nRegistration Quality: {quality}")
    print(f"RMSE: {rmse:.3f} mm")
    print(f"Inlier Ratio: {inlier_ratio*100:.1f}%")
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    save_transform(final_transform, output_dir, prefix="transformation")
    
    # Save aligned point cloud
    import open3d as o3d
    o3d.io.write_point_cloud(
        os.path.join(output_dir, "aligned_stl.ply"),
        stl_pcd_final
    )
    
    # Visualize if requested
    if visualize:
        print("Displaying visualization...")
        visualize_comparison(ct_pcd, stl_pcd, stl_pcd_final)
    
    return {
        "transformation_matrix": final_transform,
        "rmse": rmse,
        "inlier_ratio": inlier_ratio,
        "quality": quality,
        "icp_fitness": icp_result.fitness,
        "icp_inlier_rmse": icp_result.inlier_rmse
    }
