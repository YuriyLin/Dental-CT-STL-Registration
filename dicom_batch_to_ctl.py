"""
DICOM Batch to STL Converter
============================

Recursively scans directories for CT DICOM series and converts them to STL format.
Uses the same robust processing pipeline as viewer.py.

Features:
- Recursive directory scanning
- Automatic DICOM detection and CT modality filtering
- Series sorting by ImagePositionPatient / InstanceNumber
- Adaptive HU thresholding with metal artifact compensation
- mm-aware morphological cleanup
- Progress display and error logging

Usage:
    python dicom_batch_to_ctl.py --input-dir "data/CT_problem" --output-dir "output_stl"
    python dicom_batch_to_ctl.py --input-dir "data" --output-dir "stl_output" --hu-threshold auto
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import SimpleITK as sitk
import open3d as o3d
import gc  # Memory management for batch processing


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both console and file."""
    log_file = output_dir / f"conversion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    logger = logging.getLogger('dicom_batch')
    logger.setLevel(logging.DEBUG)
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler (DEBUG level for full details)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DICOM Discovery
# ============================================================================

def find_dicom_series(root_dir: Path, logger: logging.Logger) -> List[Tuple[Path, List[str]]]:
    """
    Recursively find all DICOM series in a directory.
    
    Returns:
        List of (directory_path, dicom_file_names) tuples, one per series.
    """
    logger.info(f"Scanning for DICOM series in: {root_dir}")
    series_list = []
    
    reader = sitk.ImageSeriesReader()
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dir_path = Path(dirpath)
        
        # Try to get DICOM series IDs in this directory
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dir_path))
        except Exception as e:
            logger.debug(f"Cannot read series from {dir_path}: {e}")
            continue
        
        for series_id in series_ids:
            try:
                dicom_names = reader.GetGDCMSeriesFileNames(str(dir_path), series_id)
                
                if len(dicom_names) < 10:  # Skip if too few slices
                    logger.debug(f"Skipping {dir_path} (only {len(dicom_names)} slices)")
                    continue
                
                # Check if it's CT modality
                first_file = dicom_names[0]
                file_reader = sitk.ImageFileReader()
                file_reader.SetFileName(first_file)
                file_reader.LoadPrivateTagsOn()
                file_reader.ReadImageInformation()
                
                modality = file_reader.GetMetaData("0008|0060") if file_reader.HasMetaDataKey("0008|0060") else ""
                
                if modality.upper() != "CT":
                    logger.debug(f"Skipping {dir_path}: Modality is '{modality}', not CT")
                    continue
                
                series_list.append((dir_path, list(dicom_names)))
                logger.info(f"  Found CT series: {dir_path} ({len(dicom_names)} slices)")
                
            except Exception as e:
                logger.warning(f"Error reading series {series_id} in {dir_path}: {e}")
                continue
    
    logger.info(f"Total CT series found: {len(series_list)}")
    return series_list


# ============================================================================
# CT Processing (Adapted from viewer.py)
# ============================================================================

def load_dicom_series(dicom_names: List[str], logger: logging.Logger) -> Optional[sitk.Image]:
    """Load a DICOM series from a list of file names."""
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        
        image = reader.Execute()
        
        spacing = image.GetSpacing()
        size = image.GetSize()
        
        logger.info(f"  Volume: {size}, Spacing: {spacing}")
        return image
        
    except Exception as e:
        logger.error(f"Failed to load DICOM series: {e}")
        return None


def calculate_adaptive_threshold(image: sitk.Image, logger: logging.Logger) -> int:
    """
    Robust adaptive threshold with effective spacing correction and metal detection.
    """
    logger.info("  Calculating adaptive HU threshold...")
    spacing = image.GetSpacing()
    voxel_eff = np.prod(spacing)**(1/3.0)
    array = sitk.GetArrayViewFromImage(image)
    
    # Filter out air
    roi_data = array[array > 200]
    if len(roi_data) < 1000:
        return 1200
    
    # Statistics (sampling for speed)
    sample = roi_data[::10] if len(roi_data) > 100000 else roi_data
    p95 = np.percentile(sample, 95)
    p99 = np.percentile(sample, 99)
    
    # Otsu on bone/teeth range
    bone_range = array[(array > 300) & (array < 2500)]
    if len(bone_range) > 1000:
        roi_img = sitk.GetImageFromArray(bone_range.reshape(1, 1, -1))
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.Execute(roi_img)
        otsu_val = otsu.GetThreshold()
    else:
        otsu_val = 1000

    # Hybrid Logic
    if otsu_val < 800:
        base_threshold = otsu_val * 0.4 + p95 * 0.6
    else:
        base_threshold = otsu_val
    
    # Spacing Correction
    correction = -int(200 * (voxel_eff - 0.3))
    final_threshold = int(base_threshold + correction)
    
    # Metal Artifact Heuristic - INCREASE threshold to isolate dense structure
    if p99 > 3000:
        logger.info(f"    Metal artifact detected (P99: {p99:.0f} HU), applying +150 HU offset")
        final_threshold += 150
    
    final_threshold = final_threshold - 300
    
    # Clamp to safe limits (upper limit 3000 to allow metal artifact compensation)
    final_threshold = int(np.clip(final_threshold, 500, 3000))
    
    logger.info(f"    Otsu: {otsu_val:.0f}, Voxel Eff: {voxel_eff:.3f}mm, Final: {final_threshold} HU")
    return final_threshold


def extract_surface_from_ct(
    image: sitk.Image,
    hu_threshold: int,
    smooth: bool,
    logger: logging.Logger
) -> Optional[o3d.geometry.TriangleMesh]:
    """
    Extract tooth/bone surface from CT volume.
    """
    try:
        spacing = image.GetSpacing()
        voxel_vol = spacing[0] * spacing[1] * spacing[2]
        
        # Step 1: Binary thresholding
        logger.info(f"  Segmenting at {hu_threshold} HU...")
        binary_image = sitk.BinaryThreshold(
            image,
            lowerThreshold=float(hu_threshold),
            upperThreshold=4000,
            insideValue=1,
            outsideValue=0
        )
        
        # Step 2: mm-aware Morphological cleanup (Opening to sever thin artifact connections)
        radius_mm = 0.5
        kernel_rad = [max(1, int(radius_mm / s)) for s in spacing]
        binary_image = sitk.BinaryMorphologicalOpening(binary_image, kernelRadius=kernel_rad)
        
        # Step 3: mm3-aware Connected Component Filtering
        min_vol_mm3 = 15.0
        min_size_voxels = max(10, int(min_vol_mm3 / voxel_vol))
        
        cc_filter = sitk.ConnectedComponentImageFilter()
        labeled = cc_filter.Execute(binary_image)
        relabel = sitk.RelabelComponentImageFilter()
        relabel.SetMinimumObjectSize(min_size_voxels)
        labeled = relabel.Execute(labeled)
        binary_image = sitk.BinaryThreshold(labeled, 1, 65535, 1, 0)
        
        # Step 4: Anti-aliasing
        if smooth or min(spacing) > 0.5:
            alias_rms = 0.08 * np.prod(spacing)**(1/3.0)
            marching_input = sitk.AntiAliasBinary(binary_image, maximumRMSError=alias_rms, numberOfIterations=5)
            level_val = 0.0
        else:
            marching_input = binary_image
            level_val = 0.5
        
        # Step 5: Marching cubes
        array = sitk.GetArrayFromImage(marching_input)
        origin = image.GetOrigin()
        
        from skimage import measure
        verts, faces, normals, values = measure.marching_cubes(
            array, level=level_val, spacing=spacing[::-1]
        )
        
        # Convert ZYX -> XYZ and apply origin
        verts_xyz = verts[:, [2, 1, 0]]
        verts_xyz += np.array(origin)
        
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_xyz)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        
        # Step 6: Mesh post-processing cleanup
        logger.info("  Cleaning mesh (removing artifacts)...")
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        
        # Cluster filtering: remove small disconnected mesh fragments
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # Keep only clusters with >= 100 triangles
        min_cluster_triangles = 100
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_cluster_triangles
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        
        n_removed = np.sum(triangles_to_remove)
        if n_removed > 0:
            logger.info(f"    Removed {n_removed} triangles from small clusters")
        
        logger.info(f"  Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Safety check: return None if mesh is empty
        if len(mesh.vertices) == 0:
            logger.warning("  Resulting mesh is empty! Threshold may be too high.")
            return None
        
        return mesh
        
    except Exception as e:
        logger.error(f"Surface extraction failed: {e}")
        return None


# ============================================================================
# Main Conversion Logic
# ============================================================================

def convert_series_to_stl(
    series_path: Path,
    dicom_names: List[str],
    output_dir: Path,
    hu_threshold: Optional[int],
    smooth: bool,
    logger: logging.Logger
) -> bool:
    """
    Convert a single DICOM series to STL.
    
    Returns:
        True if successful, False otherwise.
    """
    # Generate output filename from path
    # Improve: If the immediate folder is just "DICOM", use the parent (Case ID)
    p = series_path
    if p.name.upper() in ["DICOM", "DICOM_DATA", "DATA", "CT", "IMAGES"]:
        folder_name = p.parent.name.replace(" ", "_")
    else:
        # Use CaseID + FolderName if it's not generic
        case_id = p.parent.name.replace(" ", "_")
        sub_folder = p.name.replace(" ", "_")
        if case_id != sub_folder:
            folder_name = f"{case_id}_{sub_folder}"
        else:
            folder_name = sub_folder
    
    try:
        # Load DICOM
        image = load_dicom_series(dicom_names, logger)
        if image is None:
            return False
        
        # Determine threshold
        if hu_threshold is None:
            actual_hu = calculate_adaptive_threshold(image, logger)
        else:
            actual_hu = hu_threshold


        # Extract surface
        mesh = extract_surface_from_ct(image, actual_hu, smooth, logger)
        if mesh is None:
            return False
        
        # Save STL
        output_file = output_dir / f"{folder_name}_HU{actual_hu}.stl"
        o3d.io.write_triangle_mesh(str(output_file), mesh)
        logger.info(f"  Saved: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Conversion failed for {series_path}: {e}")
        return False


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    hu_threshold: Optional[int],
    smooth: bool,
    logger: logging.Logger
) -> Dict[str, int]:
    """
    Batch convert all CT DICOM series found in input directory.
    
    Returns:
        Statistics dictionary with success/failure counts.
    """
    stats = {"success": 0, "failed": 0, "total": 0}
    
    # Find all series
    series_list = find_dicom_series(input_dir, logger)
    stats["total"] = len(series_list)
    
    if not series_list:
        logger.warning("No CT DICOM series found!")
        return stats
    
    # Process each series
    for idx, (series_path, dicom_names) in enumerate(series_list, 1):
        logger.info(f"\n[{idx}/{stats['total']}] Processing: {series_path}")
        
        success = convert_series_to_stl(
            series_path, dicom_names, output_dir, hu_threshold, smooth, logger
        )
        
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        
        # Memory cleanup after each series
        gc.collect()
    
    return stats


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Batch convert CT DICOM series to STL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all CT in a directory with auto threshold
  python dicom_batch_to_ctl.py --input-dir "data/CT_problem" --output-dir "stl_output" --hu-threshold auto

  # Use fixed HU threshold
  python dicom_batch_to_ctl.py --input-dir "data" --output-dir "stl_output" --hu-threshold 1200

  # Disable anti-aliasing for sharper edges
  python dicom_batch_to_ctl.py --input-dir "data" --output-dir "stl_output" --no-smooth
        """
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Root directory to scan for DICOM files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save output STL files')
    parser.add_argument('--hu-threshold', type=str, default='auto',
                        help='HU threshold: "auto" or integer value (default: auto)')
    parser.add_argument('--no-smooth', action='store_true',
                        help='Disable anti-aliasing smoothing')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    # Determine HU threshold
    if args.hu_threshold.lower() == 'auto':
        hu_threshold = None  # Will be calculated per-series
    else:
        try:
            hu_threshold = int(args.hu_threshold)
        except ValueError:
            logger.error(f"Invalid HU threshold: {args.hu_threshold}")
            sys.exit(1)
    
    # Print banner
    logger.info("=" * 60)
    logger.info("DICOM Batch to STL Converter")
    logger.info("=" * 60)
    logger.info(f"Input:     {input_dir}")
    logger.info(f"Output:    {output_dir}")
    logger.info(f"Threshold: {'Auto' if hu_threshold is None else f'{hu_threshold} HU'}")
    logger.info(f"Smooth:    {not args.no_smooth}")
    logger.info("=" * 60)
    
    # Run batch conversion
    stats = batch_convert(
        input_dir, output_dir, hu_threshold, not args.no_smooth, logger
    )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total series found: {stats['total']}")
    logger.info(f"Successfully converted: {stats['success']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("=" * 60)
    
    if stats['failed'] > 0:
        logger.info(f"Check log file in {output_dir} for error details.")


if __name__ == '__main__':
    main()
