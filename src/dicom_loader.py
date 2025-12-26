"""
DICOM Loader Module
Handles CT DICOM series loading and tooth surface extraction.
"""

import os
import numpy as np
import SimpleITK as sitk
import open3d as o3d
from typing import Tuple


def load_dicom_series(dicom_dir: str) -> sitk.Image:
    """
    Load a complete DICOM series from a directory.
    
    Args:
        dicom_dir: Path to directory containing .dcm files
        
    Returns:
        SimpleITK.Image: 3D CT volume with proper spacing
        
    Raises:
        FileNotFoundError: If directory doesn't exist or contains no DICOM files
        RuntimeError: If DICOM series cannot be read
    """
    if not os.path.exists(dicom_dir):
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    
    # Get DICOM series using SimpleITK's ImageSeriesReader
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    
    if len(dicom_names) == 0:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")
    
    print(f"  Found {len(dicom_names)} DICOM slices")
    
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    try:
        image = reader.Execute()
    except Exception as e:
        raise RuntimeError(f"Failed to read DICOM series: {str(e)}")
    
    spacing = image.GetSpacing()
    size = image.GetSize()
    
    print(f"  Volume size: {size}")
    print(f"  Spacing: {spacing} mm")
    
    return image


def extract_teeth_surface(
    image: sitk.Image,
    hu_threshold: int = 1200,
    min_component_size: int = 100
) -> o3d.geometry.TriangleMesh:
    """
    Extract tooth surface from CT volume using HU thresholding and marching cubes.
    
    Args:
        image: SimpleITK CT volume
        hu_threshold: HU threshold for tooth segmentation (default: 1200 for enamel/dentin)
        min_component_size: Minimum voxel count for connected components (default: 100)
        
    Returns:
        open3d.geometry.TriangleMesh: Extracted tooth surface mesh
    """
    print(f"  Applying binary threshold (HU >= {hu_threshold})...")
    
    # Binary thresholding
    binary_image = sitk.BinaryThreshold(
        image,
        lowerThreshold=hu_threshold,
        upperThreshold=3000,  # Max HU value
        insideValue=1,
        outsideValue=0
    )
    
    # Morphological cleanup - remove small noise
    print("  Cleaning up binary mask...")
    binary_image = sitk.BinaryMorphologicalClosing(
        binary_image,
        kernelRadius=[1, 1, 1]
    )
    
    # Remove small connected components
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = connected_component_filter.Execute(binary_image)
    
    # Relabel to remove small components
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetMinimumObjectSize(min_component_size)
    labeled_image = relabel_filter.Execute(labeled_image)
    
    # Keep only the largest components (teeth)
    binary_image = sitk.BinaryThreshold(
        labeled_image,
        lowerThreshold=1,
        upperThreshold=255,
        insideValue=1,
        outsideValue=0
    )
    
    print("  Generating surface mesh (SimpleITK marching cubes)...")
    
    # Use SimpleITK's BinaryContour to extract surface
    contour_filter = sitk.BinaryContourImageFilter()
    contour_filter.SetFullyConnected(True)
    contour_image = contour_filter.Execute(binary_image)
    
    # Convert to mesh using marching cubes via LabelContour
    # Get points where contour exists
    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(contour_image)
    
    # Convert binary image to mesh using anti-aliasing + marching cubes approach
    # Smooth the binary mask slightly before marching cubes
    smoothed = sitk.AntiAliasBinary(
        binary_image,
        maximumRMSError=0.01,
        numberOfIterations=5
    )
    
    # Convert to array for processing with skimage marching cubes
    # (SimpleITK doesn't have direct marching cubes, so we use the smoothed distance map)
    array = sitk.GetArrayFromImage(smoothed)
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    
    # Use skimage for marching cubes (better than raw SimpleITK for this)
    from skimage import measure
    
    # Apply marching cubes - level at 0 since we're using signed distance
    try:
        verts, faces, normals, values = measure.marching_cubes(
            array,
            level=0.0,
            spacing=spacing[::-1]  # ZYX to match array ordering
        )
    except Exception as e:
        # Fallback to binary if anti-alias fails
        print(f"  Warning: Anti-alias failed ({e}), using binary image...")
        array = sitk.GetArrayFromImage(binary_image)
        verts, faces, normals, values = measure.marching_cubes(
            array,
            level=0.5,
            spacing=spacing[::-1]
        )
    
    # Convert to Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    
    # Adjust vertices from ZYX to XYZ and apply origin offset
    verts_xyz = verts[:, [2, 1, 0]]  # ZYX -> XYZ
    verts_xyz += np.array(origin)  # Apply origin offset
    
    mesh.vertices = o3d.utility.Vector3dVector(verts_xyz)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Compute vertex normals
    mesh.compute_vertex_normals()
    
    print(f"  Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh


def volume_to_pointcloud(
    mesh: o3d.geometry.TriangleMesh,
    voxel_size: float = 0.4
) -> o3d.geometry.PointCloud:
    """
    Convert mesh to downsampled point cloud with normals.
    
    Args:
        mesh: Input triangle mesh
        voxel_size: Voxel size for downsampling in mm (default: 0.4mm)
        
    Returns:
        open3d.geometry.PointCloud: Downsampled point cloud with normals
    """
    print(f"  Sampling point cloud from mesh...")
    
    # Sample points from mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=50000)
    
    print(f"  Downsampling with voxel size {voxel_size} mm...")
    
    # Voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normals
    print("  Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3,
            max_nn=30
        )
    )
    
    # Orient normals consistently (toward outside)
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    print(f"  Final point cloud: {len(pcd.points)} points")
    
    return pcd
