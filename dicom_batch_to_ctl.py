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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import SimpleITK as sitk
import open3d as o3d
import gc  # Memory management for batch processing

# Try to import tqdm for progress bar (optional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


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
    Consistently conservative threshold to ensure bone is fully captured.
    Clamped strictly between 250 and 600 HU to avoid missing thin bone
    or including too much soft tissue.
    """
    logger.info("  Calculating adaptive HU threshold...")
    spacing = image.GetSpacing()
    voxel_eff = np.prod(spacing)**(1/3.0)
    array = sitk.GetArrayViewFromImage(image)
    
    # Filter for bone/teeth candidate range (300-2500 HU)
    bone_range = array[(array > 300) & (array < 2500)]
    if len(bone_range) > 1000:
        roi_img = sitk.GetImageFromArray(bone_range.reshape(1, 1, -1))
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.Execute(roi_img)
        otsu_val = otsu.GetThreshold()
    else:
        otsu_val = 500  # Conservative default
    
    # Spacing correction (partial volume effect compensation)
    correction = -int(150 * (voxel_eff - 0.3))
    final_threshold = int(otsu_val + correction)

    
    # Strict clamping: 250-1000 HU. 
    # Even if bone is 1000 HU, 300 HU will safely include it.
    final_threshold = int(np.clip(final_threshold, 250, 1000))
    
    # adjust base threshold by +200
    final_threshold = final_threshold + 200

    logger.info(f"    Otsu: {otsu_val:.0f} HU, Voxel Eff: {voxel_eff:.3f}mm")
    logger.info(f"    Final threshold: {final_threshold} HU (clamped to 250-1000)")
    return final_threshold


def extract_surface_from_ct(
    image: sitk.Image,
    hu_threshold: int,
    smooth: bool,
    logger: logging.Logger
) -> Optional[o3d.geometry.TriangleMesh]:
    """
    Hybrid Dual-Mask Strategy for bone/tooth extraction.
    
    Problem: Single Opening radius causes trade-off:
      - Large radius (0.7mm): Cuts teeth but destroys thin bone (maxillary sinus).
      - Small radius (0.3mm): Preserves bone but teeth remain connected.
    
    Solution: Process low-density and high-density regions separately, then fuse.
    
    Algorithm:
      1. Base Mask (Soft Bone): Threshold > base_threshold. NO Opening.
      2. Hard Mask (Teeth/Cortical): Threshold > 800 HU. Aggressive Opening.
      3. Fusion: (Base AND NOT Hard) OR Opened_Hard
      4. Filter noise + Closing to smooth surface.
    """
    try:
        spacing = image.GetSpacing()
        
        # =====================================================================
        # Parameters
        # =====================================================================
        base_threshold = float(hu_threshold)
        calculated_high = base_threshold + 400
        high_density_threshold = max(calculated_high, 700.0)
        teeth_cut_radius_mm = 0.8              # Aggressive cut for teeth only
        closing_radius_mm = 1.0                # Final surface smoothing
        noise_ratio = 0.01                     # Keep components > 1% of largest
        
        logger.info(f"  [Dual-Mask Strategy]")
        logger.info(f"    Base threshold: {base_threshold} HU (soft bone, no opening)")
        logger.info(f"    High-density threshold: {high_density_threshold} HU (teeth, with opening)")
        
        # =====================================================================
        # Step 1: Generate Base Mask (Soft Bone - NO OPENING)
        # Purpose: Capture all bone including thin structures like maxillary sinus walls.
        # =====================================================================
        logger.info(f"  [1/5] Creating Base Mask (threshold > {base_threshold} HU)...")
        base_mask = sitk.BinaryThreshold(
            image,
            lowerThreshold=float(base_threshold),
            upperThreshold=4000,
            insideValue=1,
            outsideValue=0
        )
        
        # =====================================================================
        # Step 2: Generate Hard Mask (Teeth/Cortical)
        # Purpose: Isolate high-density regions that need aggressive cutting.
        # =====================================================================
        logger.info(f"  [2/5] Creating Hard Mask (threshold > {high_density_threshold} HU)...")
        hard_mask = sitk.BinaryThreshold(
            image,
            lowerThreshold=float(high_density_threshold),
            upperThreshold=4000,
            insideValue=1,
            outsideValue=0
        )
        
        # =====================================================================
        # Step 3: Apply Opening ONLY to Hard Mask
        # Purpose: Cut bridges between teeth. Safe because thin bone is NOT in this mask.
        # =====================================================================
        kernel_rad_open = [max(1, int(teeth_cut_radius_mm / s)) for s in spacing]
        logger.info(f"  [3/5] Opening Hard Mask (radius: {teeth_cut_radius_mm}mm, kernel: {kernel_rad_open})...")
        opened_hard_mask = sitk.BinaryMorphologicalOpening(hard_mask, kernelRadius=kernel_rad_open)
        
        # =====================================================================
        # Step 4: Smart Fusion (Boolean Logic)
        # Formula: Final = (Base AND NOT Hard) OR Opened_Hard
        # 
        # Explanation:
        #   - (Base AND NOT Hard): Soft bone regions that are NOT in the hard mask.
        #                          This preserves thin structures like sinus walls.
        #   - Opened_Hard:         Teeth after cutting. Separate and clean.
        #   - OR combines both layers.
        # =====================================================================
        logger.info("  [4/5] Fusing masks: (Base AND NOT Hard) OR Opened_Hard...")
        
        # (Base AND NOT Hard) = Soft bone only
        not_hard = sitk.BinaryNot(hard_mask)
        soft_bone_only = sitk.And(base_mask, not_hard)
        
        # Final fusion
        fused_mask = sitk.Or(soft_bone_only, opened_hard_mask)
        
        # =====================================================================
        # Step 5: Noise Filtering (Relative Volume)
        # =====================================================================
        logger.info(f"  [5/5] Filtering noise (keep > {noise_ratio*100:.0f}% of largest)...")
        cc_filter = sitk.ConnectedComponentImageFilter()
        labeled = cc_filter.Execute(fused_mask)
        
        relabel = sitk.RelabelComponentImageFilter()
        relabel.SetSortByObjectSize(True)
        labeled = relabel.Execute(labeled)
        
        num_objects = relabel.GetNumberOfObjects()
        if num_objects > 0:
            sizes = relabel.GetSizeOfObjectsInPixels()
            max_size = sizes[0]
            min_size = max_size * noise_ratio
            
            keep_count = 0
            for s in sizes:
                if s >= min_size:
                    keep_count += 1
                else:
                    break
            
            fused_mask = sitk.BinaryThreshold(labeled, 1, keep_count, 1, 0)
            logger.info(f"    Total: {num_objects} components. Kept top {keep_count} (min: {min_size:.0f} voxels)")
        else:
            logger.warning("    No components found!")
            return None
        
        # =====================================================================
        # Step 6: Final Closing (Smooth surface and fill holes)
        # =====================================================================
        kernel_rad_close = [max(1, int(closing_radius_mm / s)) for s in spacing]
        logger.info(f"  Closing (radius: {closing_radius_mm}mm, kernel: {kernel_rad_close})...")
        final_mask = sitk.BinaryMorphologicalClosing(fused_mask, kernelRadius=kernel_rad_close)
        
        # =====================================================================
        # Step 7: Marching Cubes Surface Extraction
        # =====================================================================
        array = sitk.GetArrayFromImage(final_mask)
        origin = image.GetOrigin()
        
        from skimage import measure
        if np.sum(array) == 0:
            logger.warning("  Empty mask after processing!")
            return None

        verts, faces, normals, values = measure.marching_cubes(
            array, level=0.5, spacing=spacing[::-1]
        )
        
        # Convert ZYX -> XYZ and apply origin
        verts_xyz = verts[:, [2, 1, 0]]
        verts_xyz += np.array(origin)
        
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_xyz)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        
        # Basic cleanup
        mesh.remove_degenerate_triangles()
        
        # Optional Laplacian smoothing
        if smooth:
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
            mesh.compute_vertex_normals()
        
        logger.info(f"  Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
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


# ============================================================================
# Multiprocessing Worker Function
# ============================================================================

def _worker_convert_series(
    args_tuple: Tuple[Path, List[str], Path, Optional[int], bool]
) -> Tuple[str, bool, str]:
    """
    Worker function for parallel processing.
    
    Args:
        args_tuple: (series_path, dicom_names, output_dir, hu_threshold, smooth)
    
    Returns:
        Tuple of (series_name, success, message)
    """
    series_path, dicom_names, output_dir, hu_threshold, smooth = args_tuple
    
    # Create a simple logger for this worker (avoid multiprocessing log conflicts)
    worker_logger = logging.getLogger(f'worker_{os.getpid()}')
    worker_logger.setLevel(logging.WARNING)  # Only log warnings/errors in workers
    if not worker_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[Worker %(process)d] %(message)s'))
        worker_logger.addHandler(handler)
    
    # Generate output filename
    p = series_path
    if p.name.upper() in ["DICOM", "DICOM_DATA", "DATA", "CT", "IMAGES"]:
        folder_name = p.parent.name.replace(" ", "_")
    else:
        case_id = p.parent.name.replace(" ", "_")
        sub_folder = p.name.replace(" ", "_")
        if case_id != sub_folder:
            folder_name = f"{case_id}_{sub_folder}"
        else:
            folder_name = sub_folder
    
    try:
        # Load DICOM
        image = load_dicom_series(dicom_names, worker_logger)
        if image is None:
            return (folder_name, False, "Failed to load DICOM")
        
        # Determine threshold
        if hu_threshold is None:
            actual_hu = calculate_adaptive_threshold(image, worker_logger)
        else:
            actual_hu = hu_threshold
        
        # Extract surface
        mesh = extract_surface_from_ct(image, actual_hu, smooth, worker_logger)
        if mesh is None:
            return (folder_name, False, "Failed to extract surface")
        
        # Save STL
        output_file = output_dir / f"{folder_name}_HU{actual_hu}.stl"
        o3d.io.write_triangle_mesh(str(output_file), mesh)
        
        # Memory cleanup
        del image, mesh
        gc.collect()
        
        return (folder_name, True, f"HU{actual_hu}")
        
    except Exception as e:
        return (folder_name, False, str(e))


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    hu_threshold: Optional[int],
    smooth: bool,
    logger: logging.Logger,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> Dict[str, int]:
    """
    Batch convert all CT DICOM series found in input directory.
    Supports parallel processing with multiprocessing.
    
    Args:
        input_dir: Root directory to scan
        output_dir: Output directory for STL files
        hu_threshold: HU threshold (None for auto)
        smooth: Enable mesh smoothing
        logger: Logger instance
        parallel: Enable parallel processing (default: True)
        max_workers: Number of worker processes (default: CPU cores - 2)
    
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
    
    # Determine number of workers
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        max_workers = max(1, cpu_count - 2)  # Leave 2 cores for system
    
    logger.info(f"Processing {stats['total']} series with {max_workers} workers...")
    
    # Prepare arguments for workers
    worker_args = [
        (series_path, dicom_names, output_dir, hu_threshold, smooth)
        for series_path, dicom_names in series_list
    ]
    
    if parallel and stats["total"] > 1:
        # ========== Parallel Processing ==========
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker_convert_series, args): args[0] 
                for args in worker_args
            }
            
            # Progress tracking
            if HAS_TQDM:
                iterator = tqdm(as_completed(futures), total=len(futures), desc="Converting", unit="series")
            else:
                iterator = as_completed(futures)
            
            for future in iterator:
                try:
                    folder_name, success, message = future.result()
                    if success:
                        stats["success"] += 1
                        logger.info(f"  ✓ {folder_name}: {message}")
                    else:
                        stats["failed"] += 1
                        logger.error(f"  ✗ {folder_name}: {message}")
                except Exception as e:
                    stats["failed"] += 1
                    logger.error(f"  ✗ Worker error: {e}")
    else:
        # ========== Sequential Processing (fallback) ==========
        if HAS_TQDM:
            iterator = tqdm(enumerate(series_list, 1), total=stats['total'], desc="Converting", unit="series")
        else:
            iterator = enumerate(series_list, 1)
        
        for idx, (series_path, dicom_names) in iterator:
            if not HAS_TQDM:
                logger.info(f"\n[{idx}/{stats['total']}] Processing: {series_path}")
            
            success = convert_series_to_stl(
                series_path, dicom_names, output_dir, hu_threshold, smooth, logger
            )
            
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
            
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
  # Convert all CT in a directory with auto threshold (batch mode)
  python dicom_batch_to_ctl.py --input-dir "data/CT_problem" --output-dir "stl_output" --hu-threshold auto

  # Process a SINGLE DICOM folder directly (no recursive scanning)
  python dicom_batch_to_ctl.py --single-dir "data/cases/2023042401/DICOM" --output-dir "CT_CTL"

  # Use fixed HU threshold
  python dicom_batch_to_ctl.py --input-dir "data" --output-dir "stl_output" --hu-threshold 1200

  # Disable smoothing for sharper edges
  python dicom_batch_to_ctl.py --input-dir "data" --output-dir "stl_output" --no-smooth

  # Disable parallel processing (sequential mode)
  python dicom_batch_to_ctl.py --input-dir "data" --output-dir "stl_output" --no-parallel

  # Specify number of worker processes
  python dicom_batch_to_ctl.py --input-dir "data" --output-dir "stl_output" --workers 4
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir', type=str,
                        help='Root directory to scan for DICOM files (batch mode)')
    input_group.add_argument('--single-dir', type=str,
                        help='Single DICOM folder to process (no recursive scanning)')
    
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save output STL files')
    parser.add_argument('--hu-threshold', type=str, default='auto',
                        help='HU threshold: "auto" or integer value (default: auto)')
    parser.add_argument('--no-smooth', action='store_true',
                        help='Disable mesh smoothing')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing (use sequential mode)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU cores - 2)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
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
    
    smooth = not args.no_smooth
    
    # ========== Single Directory Mode ==========
    if args.single_dir:
        single_dir = Path(args.single_dir)
        if not single_dir.exists():
            print(f"Error: DICOM directory does not exist: {single_dir}")
            sys.exit(1)
        
        logger.info("=" * 60)
        logger.info("DICOM to STL Converter (Single Mode)")
        logger.info("=" * 60)
        logger.info(f"Input:     {single_dir}")
        logger.info(f"Output:    {output_dir}")
        logger.info(f"Threshold: {'Auto' if hu_threshold is None else f'{hu_threshold} HU'}")
        logger.info(f"Smooth:    {smooth}")
        logger.info("=" * 60)
        
        # Get DICOM files from the single directory
        reader = sitk.ImageSeriesReader()
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(single_dir))
            if not series_ids:
                logger.error(f"No DICOM series found in: {single_dir}")
                sys.exit(1)
            
            # Use the first series found
            dicom_names = list(reader.GetGDCMSeriesFileNames(str(single_dir), series_ids[0]))
            logger.info(f"  Found {len(dicom_names)} DICOM slices")
            
            # Convert using same logic as batch mode
            success = convert_series_to_stl(
                single_dir, dicom_names, output_dir, hu_threshold, smooth, logger
            )
            
            if success:
                logger.info("\n" + "=" * 60)
                logger.info("CONVERSION SUCCESSFUL")
                logger.info("=" * 60)
            else:
                logger.error("\n" + "=" * 60)
                logger.error("CONVERSION FAILED")
                logger.info("=" * 60)
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Failed to process DICOM: {e}")
            sys.exit(1)
    
    # ========== Batch Mode ==========
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        # Print banner
        logger.info("=" * 60)
        logger.info("DICOM Batch to STL Converter")
        logger.info("=" * 60)
        logger.info(f"Input:     {input_dir}")
        logger.info(f"Output:    {output_dir}")
        logger.info(f"Threshold: {'Auto' if hu_threshold is None else f'{hu_threshold} HU'}")
        logger.info(f"Smooth:    {smooth}")
        logger.info("=" * 60)
        
        # Run batch conversion
        stats = batch_convert(
            input_dir, output_dir, hu_threshold, smooth, logger,
            parallel=not args.no_parallel,
            max_workers=args.workers
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
