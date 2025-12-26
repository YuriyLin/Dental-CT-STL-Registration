"""
Registration Module
Core registration algorithms: landmark-based SVD + ICP refinement.
"""

import json
import numpy as np
import open3d as o3d
from typing import Tuple, List
from scipy.spatial.transform import Rotation


def load_landmarks(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load landmark pairs from JSON file.
    
    Expected format:
    {
        "ct_landmarks": [[x1, y1, z1], [x2, y2, z2], ...],
        "stl_landmarks": [[x1, y1, z1], [x2, y2, z2], ...]
    }
    
    Args:
        json_path: Path to landmarks JSON file
        
    Returns:
        Tuple of (ct_landmarks, stl_landmarks) as numpy arrays (N, 3)
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If landmark format is invalid or count < 3
    """
    import os
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Landmarks file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'ct_landmarks' not in data or 'stl_landmarks' not in data:
        raise ValueError(
            "JSON must contain 'ct_landmarks' and 'stl_landmarks' keys"
        )
    
    ct_lm = np.array(data['ct_landmarks'], dtype=np.float64)
    stl_lm = np.array(data['stl_landmarks'], dtype=np.float64)
    
    if ct_lm.shape[0] < 3 or stl_lm.shape[0] < 3:
        raise ValueError(
            f"At least 3 landmark pairs required. "
            f"Found: CT={ct_lm.shape[0]}, STL={stl_lm.shape[0]}"
        )
    
    if ct_lm.shape[0] != stl_lm.shape[0]:
        raise ValueError(
            f"Landmark count mismatch: CT={ct_lm.shape[0]}, STL={stl_lm.shape[0]}"
        )
    
    if ct_lm.shape[1] != 3 or stl_lm.shape[1] != 3:
        raise ValueError(
            f"Landmarks must be 3D coordinates. "
            f"Got shapes: CT={ct_lm.shape}, STL={stl_lm.shape}"
        )
    
    print(f"  Loaded {ct_lm.shape[0]} landmark pairs")
    
    return ct_lm, stl_lm


def compute_rigid_transform_svd(
    source_points: np.ndarray,
    target_points: np.ndarray
) -> np.ndarray:
    """
    Compute rigid transformation using SVD (Procrustes analysis).
    
    Explicitly checks for reflection and corrects it to ensure proper rotation.
    
    Args:
        source_points: Source landmarks (N, 3)
        target_points: Target landmarks (N, 3)
        
    Returns:
        4x4 transformation matrix (homogeneous coordinates)
        
    Algorithm:
        1. Center both point sets
        2. Compute optimal rotation using SVD
        3. Check det(R) - if -1 (reflection), flip last column of V
        4. Compute translation
    """
    # Ensure float64 for numerical stability
    source = source_points.astype(np.float64)
    target = target_points.astype(np.float64)
    
    # Compute centroids
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    
    # Center the points
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    
    # Compute cross-covariance matrix
    H = source_centered.T @ target_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Check for reflection and correct it
    det_R = np.linalg.det(R)
    
    if det_R < 0:
        print("  Warning: SVD produced reflection (det(R) = -1), correcting...")
        # Flip the last column of Vt (last row after transpose)
        Vt_corrected = Vt.copy()
        Vt_corrected[-1, :] *= -1
        R = Vt_corrected.T @ U.T
        
        # Verify correction
        det_R_corrected = np.linalg.det(R)
        print(f"  After correction: det(R) = {det_R_corrected:.6f}")
    
    # Compute translation
    t = target_centroid - R @ source_centroid
    
    # Construct 4x4 transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    
    # Compute and report landmark alignment error
    source_transformed = (R @ source.T).T + t
    errors = np.linalg.norm(source_transformed - target, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    print(f"  Landmark alignment RMSE: {rmse:.3f} mm")
    print(f"  Landmark errors - Min: {errors.min():.3f}, Max: {errors.max():.3f}, Mean: {errors.mean():.3f} mm")
    
    return T


def apply_transform(
    pcd: o3d.geometry.PointCloud,
    transform: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    Apply rigid transformation to point cloud.
    
    Args:
        pcd: Input point cloud
        transform: 4x4 transformation matrix
        
    Returns:
        Transformed point cloud (new copy)
    """
    pcd_transformed = o3d.geometry.PointCloud(pcd)
    pcd_transformed.transform(transform)
    return pcd_transformed


def refine_with_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    max_iterations_stage1: int = 30,
    max_iterations_stage2: int = 50
) -> Tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult]:
    """
    Refine alignment using two-stage ICP.
    
    Stage 1: max_dist = 2.0mm (tolerance for initial alignment)
    Stage 2: max_dist = 1.0mm (tight refinement)
    
    Args:
        source: Source point cloud (already pre-aligned)
        target: Target point cloud
        initial_transform: Initial transformation matrix from landmark registration
        max_iterations_stage1: Max iterations for stage 1 (default: 30)
        max_iterations_stage2: Max iterations for stage 2 (default: 50)
        
    Returns:
        Tuple of (final_transform, registration_result)
    """
    print("\n  === Stage 1 ICP: max_dist = 2.0mm ===")
    
    # Stage 1: Coarse refinement with 2.0mm threshold
    result_stage1 = o3d.pipelines.registration.registration_icp(
        source=source,
        target=target,
        max_correspondence_distance=2.0,
        init=np.eye(4),  # Source is already aligned, so identity
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations_stage1,
            relative_fitness=1e-6,
            relative_rmse=1e-6
        )
    )
    
    print(f"  Fitness: {result_stage1.fitness:.4f}")
    print(f"  Inlier RMSE: {result_stage1.inlier_rmse:.3f} mm")
    
    # Apply stage 1 result
    intermediate_transform = initial_transform @ result_stage1.transformation
    source_stage1 = apply_transform(source, result_stage1.transformation)
    
    print("\n  === Stage 2 ICP: max_dist = 1.0mm ===")
    
    # Stage 2: Fine refinement with 1.0mm threshold
    result_stage2 = o3d.pipelines.registration.registration_icp(
        source=source_stage1,
        target=target,
        max_correspondence_distance=1.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations_stage2,
            relative_fitness=1e-6,
            relative_rmse=1e-6
        )
    )
    
    print(f"  Fitness: {result_stage2.fitness:.4f}")
    print(f"  Inlier RMSE: {result_stage2.inlier_rmse:.3f} mm")
    
    # Combine transformations
    final_transform = intermediate_transform @ result_stage2.transformation
    
    return final_transform, result_stage2
