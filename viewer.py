"""
Point Cloud Viewer for Dental CT-STL Registration
==================================================

A simple command-line tool to visualize CT (DICOM) and STL data as point clouds.
Useful for quality inspection before running the registration pipeline.

Features:
- CT (DICOM) → Surface extraction → Point cloud (Red)
- STL → Uniform sampling → Point cloud (Green)
- Can display both simultaneously for visual comparison

Usage Examples:
    # View CT only
    python viewer.py --dicom-dir "data/CASE/cases/2023041102/DICOM" --mode ct

    # View STL only
    python viewer.py --stl-path "data/CASE/cases/2023041102/mandible/gum.stl" --mode stl

    # View both CT and STL
    python viewer.py --dicom-dir "data/CASE/cases/2023041102/DICOM" \
                     --stl-path "data/CASE/cases/2023041102/mandible/gum.stl" \
                     --mode both

    # Custom parameters
    python viewer.py --dicom-dir "data/CT_problem/metal artifact" \
                     --mode ct --hu-threshold 1100 --ct-voxel-size 0.5
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import open3d as o3d
import SimpleITK as sitk


# ============================================================================
# CT (DICOM) Processing Functions
# ============================================================================

def load_dicom_series(dicom_dir: str) -> sitk.Image:
    """
    Load a complete DICOM series from a directory.
    
    Args:
        dicom_dir: Path to directory containing .dcm files
        
    Returns:
        SimpleITK.Image: 3D CT volume with proper spacing
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    
    if len(dicom_names) == 0:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")
    
    print(f"  Found {len(dicom_names)} DICOM slices")
    
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    image = reader.Execute()
    
    spacing = image.GetSpacing()
    size = image.GetSize()
    
    print(f"  Volume size: {size}")
    print(f"  Spacing: {spacing} mm")
    
    return image



def calculate_adaptive_threshold(image: sitk.Image) -> int:
    """
    Robust adaptive threshold with effective spacing correction and metal detection.
    """
    print("  Calculating robust adaptive threshold...")
    spacing = image.GetSpacing()
    voxel_eff = np.prod(spacing)**(1/3.0)  # Effective spacing based on volume
    array = sitk.GetArrayViewFromImage(image)
    
    # 1. Filter out air
    roi_data = array[array > 200]
    if len(roi_data) < 1000:
        return 1200
    
    # 2. Statistics (sample for speed)
    sample = roi_data[::10] if len(roi_data) > 100000 else roi_data
    p95 = np.percentile(sample, 95)
    p99 = np.percentile(sample, 99)
    
    # 3. Otsu on bone/teeth range
    bone_range = array[(array > 300) & (array < 2500)]
    if len(bone_range) > 1000:
        roi_img = sitk.GetImageFromArray(bone_range.reshape(1, 1, -1))
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.Execute(roi_img)
        otsu_val = otsu.GetThreshold()
    else:
        otsu_val = 1000

    # 4. Hybrid Logic
    if otsu_val < 800:
        base_threshold = otsu_val * 0.4 + p95 * 0.6
    else:
        base_threshold = otsu_val
    
    # 5. Spacing Correction (Effective spacing based)
    correction = -int(200 * (voxel_eff - 0.3))
    final_threshold = int(base_threshold + correction)
    
    # 6. Metal Artifact Heuristic
    if p99 > 3000:
        print(f"   Metal artifact suspected (P99: {p99:.0f} HU > 3000), applying safety offset -80 HU")
        final_threshold -= 80
    
    # Adding user-requested offset (+200 HU)
    final_threshold += 200
    
    # Clamp to safe limits
    final_threshold = int(np.clip(final_threshold, 500, 1800))
    
    print(f"    - Bone Otsu: {otsu_val:.1f} HU")
    print(f"    - Voxel Eff: {voxel_eff:.3f} mm")
    print(f"    - Final robust threshold: {final_threshold} HU")
    
    return final_threshold


def extract_surface_from_ct(
    image: sitk.Image,
    hu_threshold: int,
    smooth: bool = True
) -> o3d.geometry.TriangleMesh:
    """
    Extract surface with mm3-aware filtering and spacing-linked RMS.
    """
    spacing = image.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    
    # Step 1: Binary thresholding
    binary_image = sitk.BinaryThreshold(
        image,
        lowerThreshold=float(hu_threshold),
        upperThreshold=4000,
        insideValue=1,
        outsideValue=0
    )
    
    # Step 2: mm-aware Morphological cleanup (0.5mm radius closing)
    radius_mm = 0.5
    kernel_rad = [max(1, int(radius_mm / s)) for s in spacing]
    binary_image = sitk.BinaryMorphologicalClosing(binary_image, kernelRadius=kernel_rad)
    
    # Step 3: mm3-aware Connected Component Filtering
    # Filter out anything smaller than 15 mm3 (small noise)
    min_vol_mm3 = 15.0
    min_size_voxels = max(10, int(min_vol_mm3 / voxel_vol))
    print(f"  Cleaning binary mask (Volume filter: {min_vol_mm3}mm³, min_voxels: {min_size_voxels})...")
    
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled = cc_filter.Execute(binary_image)
    relabel = sitk.RelabelComponentImageFilter()
    relabel.SetMinimumObjectSize(min_size_voxels)
    labeled = relabel.Execute(labeled)
    binary_image = sitk.BinaryThreshold(labeled, 1, 65535, 1, 0)
    
    # Step 4: Optional Anti-alias with spacing-linked RMS
    # Mandatory for large spacing (> 0.5mm)
    if smooth or min(spacing) > 0.5:
        if not smooth:
            print("  [Force Enable] Anti-aliasing mandated by large spacing (>0.5mm)")
        # RMS tied to spacing: tighter on high-res, looser on low-res
        alias_rms = 0.08 * np.prod(spacing)**(1/3.0) 
        print(f"  Applying Anti-alias smoothing (RMS: {alias_rms:.4f} pixels)...")
        marching_input = sitk.AntiAliasBinary(binary_image, maximumRMSError=alias_rms, numberOfIterations=5)
        level_val = 0.0
    else:
        print("  Skipping anti-aliasing (using raw binary)...")
        marching_input = binary_image
        level_val = 0.5
    
    # Step 4: Marching cubes
    array = sitk.GetArrayFromImage(marching_input)
    origin = image.GetOrigin()
    
    from skimage import measure
    try:
        verts, faces, normals, values = measure.marching_cubes(
            array, level=level_val, spacing=spacing[::-1]  # ZYX ordering
        )
    except Exception:
        # Fallback to binary
        array = sitk.GetArrayFromImage(binary_image)
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
    
    print(f"  Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh


def mesh_to_pointcloud(mesh: o3d.geometry.TriangleMesh, voxel_size: float) -> o3d.geometry.PointCloud:
    """
    Convert mesh surface to uniformly sampled point cloud.
    
    Args:
        mesh: Input triangle mesh
        voxel_size: Voxel size for downsampling (mm)
        
    Returns:
        Downsampled point cloud with normals
    """
    # Sample points from surface (not volume!)
    pcd = mesh.sample_points_uniformly(number_of_points=100000)
    
    # Voxel downsampling for uniform density
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
    )
    
    return pcd


# ============================================================================
# STL Processing Functions
# ============================================================================

def load_stl_as_pointcloud(stl_path: str, num_points: int) -> o3d.geometry.PointCloud:
    """
    Load STL file and convert to uniformly sampled point cloud.
    
    Args:
        stl_path: Path to STL file
        num_points: Number of points to sample
        
    Returns:
        Point cloud with normals
    """
    print(f"  Loading STL: {Path(stl_path).name}")
    mesh = o3d.io.read_triangle_mesh(stl_path)
    
    if not mesh.has_vertices():
        raise RuntimeError(f"Failed to load mesh from {stl_path}")
    
    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Uniform sampling from surface
    print(f"  Sampling {num_points} points...")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    
    return pcd


# ============================================================================
# Main Viewer
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Point Cloud Viewer for CT (DICOM) and STL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View CT only
  python viewer.py --dicom-dir "data/CASE/cases/2023041102/DICOM" --mode ct

  # View STL only  
  python viewer.py --stl-path "data/CASE/cases/2023041102/mandible/gum.stl" --mode stl

  # View both
  python viewer.py --dicom-dir "data/CASE/cases/2023041102/DICOM" \\
                   --stl-path "data/CASE/cases/2023041102/mandible/gum.stl" \\
                   --mode both

  # Metal artifact case with custom HU
  python viewer.py --dicom-dir "data/CT_problem/metal artifact" --mode ct --hu-threshold 1100
        """
    )
    
    # Input paths
    parser.add_argument('--dicom-dir', type=str, 
                        help='Path to DICOM folder (required for ct/both mode)')
    parser.add_argument('--stl-path', type=str,
                        help='Path to STL file (required for stl/both mode)')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['ct', 'stl', 'both'], default='ct',
                        help='What to view: ct, stl, or both (default: ct)')
    
    # Processing parameters
    parser.add_argument('--hu-threshold', type=str, default='1200',
                        help='HU threshold for CT segmentation. Use "auto" for adaptive threshold (default: 1200)')
    parser.add_argument('--ct-voxel-size', type=float, default=0.4,
                        help='Voxel size for CT point cloud sampling in mm (default: 0.4)')
    parser.add_argument('--stl-points', type=int, default=10000,
                        help='Number of points to sample from STL (default: 10000)')
    parser.add_argument('--no-smooth', action='store_true',
                        help='Disable anti-alias smoothing for CT surface extraction')
    
    # Output options
    parser.add_argument('--output-stl', type=str,
                        help='Path to save extracted CT surface as STL file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.mode in ['ct', 'both'] and not args.dicom_dir:
        print("Error: --dicom-dir is required for ct/both mode")
        sys.exit(1)
    if args.mode in ['stl', 'both'] and not args.stl_path:
        print("Error: --stl-path is required for stl/both mode")
        sys.exit(1)
    
    geometries = []
    
    # ========== Process CT ==========
    if args.mode in ['ct', 'both']:
        print("\n" + "=" * 50)
        print("Processing CT (DICOM)")
        print("=" * 50)
        
        ct_image = load_dicom_series(args.dicom_dir)
        
        # 1. Determine actual threshold (Calculate ONCE)
        if args.hu_threshold.lower() == 'auto':
            actual_hu = calculate_adaptive_threshold(ct_image)
        else:
            actual_hu = int(args.hu_threshold)
            
        print(f"  Selected Configuration:")
        voxel_eff = np.prod(ct_image.GetSpacing())**(1/3.0)
        print(f"    - Voxel Eff: {voxel_eff:.3f} mm")
        print(f"    - Threshold: {actual_hu} HU")
        
        # 2. Extract mesh
        ct_mesh = extract_surface_from_ct(
            ct_image, 
            hu_threshold=actual_hu, 
            smooth=(not args.no_smooth)
        )
        
        # 3. Create point cloud
        ct_pcd = mesh_to_pointcloud(ct_mesh, voxel_size=args.ct_voxel_size)
        
        # Color: Red
        ct_pcd.paint_uniform_color([1.0, 0.3, 0.3])
        geometries.append(ct_pcd)
        
        print(f"  Final CT point cloud: {len(ct_pcd.points)} points (Red)")
        
        # Save STL if requested
        if args.output_stl:
            # Use smart filename logic (same as dicom_batch_to_ctl.py)
            p = Path(args.dicom_dir)
            if p.name.upper() in ["DICOM", "DICOM_DATA", "DATA", "CT", "IMAGES"]:
                folder_name = p.parent.name.replace(" ", "_")
            else:
                case_id = p.parent.name.replace(" ", "_")
                sub_folder = p.name.replace(" ", "_")
                if case_id != sub_folder:
                    folder_name = f"{case_id}_{sub_folder}"
                else:
                    folder_name = sub_folder
                    
            default_name = f"{folder_name}_HU{actual_hu}.stl"
            
            # Use CT_CTL as default directory if none specified or if it's a relative path
            output_val = args.output_stl
            output_path = Path(output_val)  
            
            if output_path.is_dir() or output_val.lower() == 'true' or output_val == '':
                # Always ensure CT_CTL exists
                target_dir = Path("CT_CTL")
                target_dir.mkdir(parents=True, exist_ok=True)
                save_path = target_dir / default_name
            else:
                # If a specific filename path is provided, use it
                save_path = output_path
                # Ensure parent directory exists
                save_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  Saving STL to: {save_path} (Overwriting if exists)")
            o3d.io.write_triangle_mesh(str(save_path), ct_mesh)
            print(f"  ✓ STL saved successfully!")
    
    # ========== Process STL ==========
    if args.mode in ['stl', 'both']:
        print("\n" + "=" * 50)
        print("Processing STL")
        print("=" * 50)
        
        stl_pcd = load_stl_as_pointcloud(args.stl_path, num_points=args.stl_points)
        
        # Color: Green
        stl_pcd.paint_uniform_color([0.3, 1.0, 0.3])
        geometries.append(stl_pcd)
        
        print(f"  Final STL point cloud: {len(stl_pcd.points)} points (Green)")
    
    # ========== Open Viewer ==========
    print("\n" + "=" * 50)
    print("Opening Viewer")
    print("=" * 50)
    print("Controls:")
    print("  Left-click drag  : Rotate")
    print("  Scroll wheel     : Zoom")
    print("  Right-click drag : Pan")
    print("  Q                : Quit")
    print("=" * 50)
    
    window_title = f"Point Cloud Viewer - Mode: {args.mode}"
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_title,
        width=1280,
        height=720
    )
    
    print("\nViewer closed.")


if __name__ == '__main__':
    main()
