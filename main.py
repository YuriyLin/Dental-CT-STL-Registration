"""
Dental CT-STL Registration Pipeline
Main entry point with CLI interface.

Clinical-grade registration system for aligning intraoral STL scans 
to CT coordinate systems for dental implant and surgical planning.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.dicom_loader import load_dicom_series, extract_teeth_surface, volume_to_pointcloud
from src.stl_loader import load_stl, prepare_pointcloud_for_icp
from src.registration import (
    load_landmarks,
    compute_rigid_transform_svd,
    apply_transform,
    refine_with_icp
)
from src.visualizer import visualize_comparison
from src.utils import (
    compute_rmse,
    compute_inlier_ratio,
    evaluate_registration_quality,
    save_transform,
    print_registration_report
)

import open3d as o3d


class InteractiveLandmarkPicker:
    """Interactive landmark picker integrated into pipeline."""
    
    def __init__(self, ct_pcd, stl_pcd, case_name):
        self.ct_pcd = ct_pcd
        self.stl_pcd = stl_pcd
        self.case_name = case_name
        self.ct_landmarks = []
        self.stl_landmarks = []
    
    def pick_points(self, pcd, title, color, num_points=None):
        """
        Interactive point picking.
        
        Instructions:
        - Shift + Left Click: Pick a point
        - Press 'Q': Finish and close window
        """
        print(f"\n{'='*60}")
        print(f"PICKING LANDMARKS: {title}")
        print(f"{'='*60}")
        if num_points:
            print(f"Pick exactly {num_points} landmarks (in same order as CT)")
        else:
            print("Pick at least 3 landmarks")
        print("Instructions:")
        print("  1. Rotate/Pan/Zoom to find anatomical landmark")
        print("  2. SHIFT + LEFT CLICK to pick a point")
        print("  3. Press 'Q' when finished")
        print(f"{'='*60}\n")
        
        pcd_vis = o3d.geometry.PointCloud(pcd)
        pcd_vis.paint_uniform_color(color)
        
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(pcd_vis)
        vis.run()
        
        picked_indices = vis.get_picked_points()
        vis.destroy_window()
        
        if len(picked_indices) == 0:
            return []
        
        points_array = np.asarray(pcd.points)
        picked_coords = [points_array[idx].tolist() for idx in picked_indices]
        
        print(f"\n✓ Picked {len(picked_coords)} landmarks:")
        for i, pt in enumerate(picked_coords):
            print(f"  {i+1}. [{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]")
        
        return picked_coords
    
    def run(self):
        """Run interactive landmark picking."""
        print("\n" + "="*60)
        print("INTERACTIVE LANDMARK PICKER")
        print("="*60)
        print(f"Case: {self.case_name}")
        print("Recommended landmarks:")
        print("  - Central incisor incisal edge (11 or 21)")
        print("  - Left first molar mesiobuccal cusp (26 or 36)")
        print("  - Right first molar mesiobuccal cusp (16 or 46)")
        print("="*60)
        
        # Pick CT landmarks
        self.ct_landmarks = self.pick_points(
            self.ct_pcd,
            f"CT - {self.case_name}",
            [1.0, 0.4, 0.4]
        )
        
        if len(self.ct_landmarks) < 3:
            print("\n❌ Error: Need at least 3 landmarks on CT!")
            return False
        
        # Pick STL landmarks
        self.stl_landmarks = self.pick_points(
            self.stl_pcd,
            f"STL - {self.case_name} (Pick {len(self.ct_landmarks)} corresponding points)",
            [0.4, 1.0, 0.4],
            num_points=len(self.ct_landmarks)
        )
        
        if len(self.stl_landmarks) != len(self.ct_landmarks):
            print(f"\n❌ Error: Landmark count mismatch!")
            print(f"  CT: {len(self.ct_landmarks)}, STL: {len(self.stl_landmarks)}")
            return False
        
        return True
    
    def save(self, output_path):
        """Save landmarks to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            "ct_landmarks": self.ct_landmarks,
            "stl_landmarks": self.stl_landmarks,
            "case_name": self.case_name,
            "num_landmarks": len(self.ct_landmarks)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Landmarks saved to: {output_path}")


def find_stl_file(mandible_dir: str) -> str:
    """
    Find STL file in mandible directory (supports any .stl file).
    
    Args:
        mandible_dir: Directory containing STL file
        
    Returns:
        Path to STL file
        
    Raises:
        FileNotFoundError: If no STL file found
    """
    stl_files = list(Path(mandible_dir).glob('*.stl'))
    
    if len(stl_files) == 0:
        raise FileNotFoundError(f"No STL file found in {mandible_dir}")
    
    if len(stl_files) > 1:
        print(f"Warning: Multiple STL files found, using first: {stl_files[0].name}")
    
    return str(stl_files[0])


def main():
    parser = argparse.ArgumentParser(
        description='Dental CT-STL Registration Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # With existing landmarks:
  python main.py --case 2023041102 --landmarks landmarks/2023041102.json --visualize
  
  # Interactive mode (pick landmarks if file doesn't exist):
  python main.py --case 2023041102 --landmarks landmarks/2023041102.json --interactive --visualize
  
  # Custom HU threshold:
  python main.py --case 2023041102 --landmarks landmarks/2023041102.json --hu-threshold 1100
        """
    )
    
    parser.add_argument(
        '--case',
        type=str,
        required=True,
        help='Case name (subdirectory in data/CASE/cases/)'
    )
    
    parser.add_argument(
        '--landmarks',
        type=str,
        required=True,
        help='Path to landmarks JSON file'
    )
    
    parser.add_argument(
        '--hu-threshold',
        type=int,
        default=1200,
        help='HU threshold for tooth extraction (default: 1200)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='processed',
        help='Output directory for results (default: processed)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization windows'
    )
    
    parser.add_argument(
        '--no-icp',
        action='store_true',
        help='Skip ICP refinement (landmark-only registration)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Launch interactive landmark picker if landmarks file does not exist'
    )
    
    args = parser.parse_args()
    
    # Construct paths
    case_dir = Path('data/CASE/cases') / args.case
    dicom_dir = case_dir / 'DICOM'
    mandible_dir = case_dir / 'mandible'
    output_dir = Path(args.output_dir) / args.case
    
    # Validate inputs
    if not dicom_dir.exists():
        print(f"Error: DICOM directory not found: {dicom_dir}")
        sys.exit(1)
    
    if not mandible_dir.exists():
        print(f"Error: Mandible directory not found: {mandible_dir}")
        sys.exit(1)
    
    # Handle landmarks file
    landmarks_path = Path(args.landmarks)
    need_interactive_picking = False
    
    if not landmarks_path.exists():
        if args.interactive:
            print(f"Landmarks file not found: {args.landmarks}")
            print("Launching interactive landmark picker...")
            need_interactive_picking = True
        else:
            print(f"Error: Landmarks file not found: {args.landmarks}")
            print("Use --interactive flag to launch landmark picker")
            sys.exit(1)
    
    # Find STL file
    try:
        stl_path = find_stl_file(str(mandible_dir))
        print(f"Found STL file: {Path(stl_path).name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("DENTAL CT-STL REGISTRATION PIPELINE")
    print("=" * 60)
    print(f"Case:           {args.case}")
    print(f"DICOM dir:      {dicom_dir}")
    print(f"STL file:       {stl_path}")
    print(f"Landmarks:      {args.landmarks}")
    print(f"HU threshold:   {args.hu_threshold}")
    print(f"Output dir:     {output_dir}")
    print("=" * 60 + "\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== STAGE 1: Load CT and extract teeth ==========
    print("[1/7] Loading CT DICOM series...")
    ct_image = load_dicom_series(str(dicom_dir))
    
    print(f"\n[2/7] Extracting teeth surface (HU threshold: {args.hu_threshold})...")
    ct_mesh = extract_teeth_surface(ct_image, hu_threshold=args.hu_threshold)
    
    print("\n[3/7] Converting CT mesh to point cloud...")
    ct_pcd = volume_to_pointcloud(ct_mesh, voxel_size=0.4)
    
    # Save CT point cloud for reference
    o3d.io.write_point_cloud(str(output_dir / "ct_teeth.ply"), ct_pcd)
    
    # ========== STAGE 2: Load STL ==========
    print(f"\n[4/7] Loading STL file...")
    stl_mesh = load_stl(stl_path)
    
    print("\n[4/7] Preparing STL point cloud...")
    stl_pcd = prepare_pointcloud_for_icp(stl_mesh, target_points=8000)
    
    # ========== INTERACTIVE LANDMARK PICKING (if needed) ==========
    if need_interactive_picking:
        picker = InteractiveLandmarkPicker(ct_pcd, stl_pcd, args.case)
        success = picker.run()
        
        if not success:
            print("\n❌ Landmark picking failed or cancelled.")
            sys.exit(1)
        
        picker.save(str(landmarks_path))
        print("\nProceeding with registration...")
    
    # ========== STAGE 3: Landmark-based registration ==========
    print(f"\n[5/7] Loading landmarks...")
    ct_landmarks, stl_landmarks = load_landmarks(args.landmarks)
    landmark_count = len(ct_landmarks)
    
    print("\n[5/7] Computing rigid transformation (SVD)...")
    initial_transform = compute_rigid_transform_svd(stl_landmarks, ct_landmarks)
    
    print("\n[5/7] Applying initial transformation...")
    stl_pcd_aligned = apply_transform(stl_pcd, initial_transform)
    
    # ========== STAGE 4: ICP refinement (optional) ==========
    if not args.no_icp:
        print(f"\n[6/7] Refining with two-stage ICP...")
        final_transform, icp_result = refine_with_icp(
            stl_pcd_aligned, ct_pcd, initial_transform
        )
        icp_fitness = icp_result.fitness
        icp_rmse = icp_result.inlier_rmse
    else:
        print("\n[6/7] Skipping ICP refinement (--no-icp flag set)")
        final_transform = initial_transform
        icp_fitness = 0.0
        icp_rmse = 0.0
    
    # Apply final transform
    stl_pcd_final = apply_transform(stl_pcd, final_transform)
    
    # ========== STAGE 5: Quality evaluation ==========
    print(f"\n[7/7] Evaluating registration quality...")
    rmse = compute_rmse(stl_pcd_final, ct_pcd)
    inlier_ratio = compute_inlier_ratio(stl_pcd_final, ct_pcd, threshold=1.0)
    quality = evaluate_registration_quality(rmse, inlier_ratio)
    
    # Print report
    print_registration_report(
        rmse=rmse,
        inlier_ratio=inlier_ratio,
        quality=quality,
        landmark_count=landmark_count,
        icp_fitness=icp_fitness,
        icp_rmse=icp_rmse
    )
    
    # ========== STAGE 6: Save results ==========
    print("Saving results...")
    save_transform(final_transform, str(output_dir), prefix="transformation")
    
    # Save aligned STL point cloud
    o3d.io.write_point_cloud(str(output_dir / "aligned_stl.ply"), stl_pcd_final)
    print(f"  Saved aligned STL: {output_dir / 'aligned_stl.ply'}")
    
    # Save original STL for reference
    o3d.io.write_point_cloud(str(output_dir / "original_stl.ply"), stl_pcd)
    print(f"  Saved original STL: {output_dir / 'original_stl.ply'}")
    
    # ========== STAGE 7: Visualization ==========
    if args.visualize:
        visualize_comparison(ct_pcd, stl_pcd, stl_pcd_final)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
