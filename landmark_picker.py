"""
Interactive Landmark Picker Tool
Manually select corresponding landmarks on CT and STL for registration.
"""

import os
import sys
import json
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.dicom_loader import load_dicom_series, extract_teeth_surface, volume_to_pointcloud
from src.stl_loader import load_stl, prepare_pointcloud_for_icp
from src.registration import compute_rigid_transform_svd


class LandmarkPicker:
    """Interactive landmark picker for CT and STL point clouds."""
    
    def __init__(self, ct_pcd, stl_pcd, case_name):
        self.ct_pcd = ct_pcd
        self.stl_pcd = stl_pcd
        self.case_name = case_name
        
        self.ct_landmarks = []
        self.stl_landmarks = []
        
        self.current_mode = "ct"  # "ct" or "stl"
        self.picked_points = []
        
    def pick_points_interactive(self, pcd, title, color):
        """
        Interactive point picking using Open3D visualizer.
        
        User Instructions:
        - Shift + Left Click: Pick a point
        - Press 'Q': Finish picking and close window
        """
        print(f"\n{'='*60}")
        print(f"PICKING LANDMARKS ON: {title}")
        print(f"{'='*60}")
        print("Instructions:")
        print("  1. Rotate/Pan/Zoom to find anatomical landmark")
        print("  2. SHIFT + LEFT CLICK to pick a point")
        print("  3. Press 'Q' when finished (minimum 3 points)")
        print(f"{'='*60}\n")
        
        # Create a copy and color it
        pcd_vis = o3d.geometry.PointCloud(pcd)
        pcd_vis.paint_uniform_color(color)
        
        picked_points = []
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(pcd_vis)
        
        # Run the visualizer
        vis.run()
        
        # Get picked points
        picked_indices = vis.get_picked_points()
        vis.destroy_window()
        
        if len(picked_indices) == 0:
            print("  Warning: No points picked!")
            return []
        
        # Convert indices to coordinates
        points_array = np.asarray(pcd.points)
        picked_coords = [points_array[idx].tolist() for idx in picked_indices]
        
        print(f"\n  ✓ Picked {len(picked_coords)} landmarks:")
        for i, pt in enumerate(picked_coords):
            print(f"    {i+1}. [{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]")
        
        return picked_coords
    
    def run(self):
        """Run the interactive landmark picking process."""
        
        print("\n" + "="*60)
        print("INTERACTIVE LANDMARK PICKER")
        print("="*60)
        print(f"Case: {self.case_name}")
        print("="*60)
        
        # Step 1: Pick CT landmarks
        print("\n[STEP 1/2] Pick landmarks on CT...")
        self.ct_landmarks = self.pick_points_interactive(
            self.ct_pcd,
            f"CT - Pick Landmarks (Case: {self.case_name})",
            [1.0, 0.4, 0.4]  # Red
        )
        
        if len(self.ct_landmarks) < 3:
            print("\n❌ Error: Need at least 3 landmarks on CT!")
            return False
        
        # Step 2: Pick STL landmarks (same number)
        print(f"\n[STEP 2/2] Pick {len(self.ct_landmarks)} CORRESPONDING landmarks on STL...")
        print(f"  (Pick in the SAME ORDER as CT)")
        
        self.stl_landmarks = self.pick_points_interactive(
            self.stl_pcd,
            f"STL - Pick {len(self.ct_landmarks)} Corresponding Landmarks",
            [0.4, 1.0, 0.4]  # Green
        )
        
        if len(self.stl_landmarks) != len(self.ct_landmarks):
            print(f"\n❌ Error: Landmark count mismatch!")
            print(f"  CT: {len(self.ct_landmarks)} landmarks")
            print(f"  STL: {len(self.stl_landmarks)} landmarks")
            print(f"  Please pick exactly {len(self.ct_landmarks)} landmarks on STL.")
            return False
        
        return True
    
    def save_landmarks(self, output_path):
        """Save landmarks to JSON file."""
        
        landmarks_data = {
            "ct_landmarks": self.ct_landmarks,
            "stl_landmarks": self.stl_landmarks,
            "case_name": self.case_name,
            "num_landmarks": len(self.ct_landmarks)
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(landmarks_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Landmarks saved to: {output_path}")
    
    def compute_and_preview_transform(self):
        """Compute SVD transform and show preview."""
        
        print("\n" + "="*60)
        print("COMPUTING SVD TRANSFORMATION")
        print("="*60)
        
        ct_pts = np.array(self.ct_landmarks)
        stl_pts = np.array(self.stl_landmarks)
        
        # Compute SVD transformation
        transform = compute_rigid_transform_svd(stl_pts, ct_pts)
        
        # Apply transform to STL
        stl_aligned = o3d.geometry.PointCloud(self.stl_pcd)
        stl_aligned.transform(transform)
        
        # Visualize result
        print("\nShowing alignment preview...")
        print("  Red: CT")
        print("  Green: Aligned STL")
        print("  Press Q to close")
        
        ct_vis = o3d.geometry.PointCloud(self.ct_pcd)
        ct_vis.paint_uniform_color([1.0, 0.4, 0.4])
        
        stl_vis = o3d.geometry.PointCloud(stl_aligned)
        stl_vis.paint_uniform_color([0.4, 1.0, 0.4])
        
        o3d.visualization.draw_geometries(
            [ct_vis, stl_vis],
            window_name="Landmark-based Alignment Preview",
            width=1280,
            height=720
        )


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Landmark Picker for CT-STL Registration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python landmark_picker.py --case 2023041102
  python landmark_picker.py --case 2023041102 --hu-threshold 1100 --preview
        """
    )
    
    parser.add_argument(
        '--case',
        type=str,
        required=True,
        help='Case name (subdirectory in data/CASE/cases/)'
    )
    
    parser.add_argument(
        '--hu-threshold',
        type=int,
        default=1200,
        help='HU threshold for tooth extraction (default: 1200)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON path (default: landmarks/{case_name}.json)'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show alignment preview after picking landmarks'
    )
    
    args = parser.parse_args()
    
    # Construct paths
    case_dir = Path('data/CASE/cases') / args.case
    dicom_dir = case_dir / 'DICOM'
    mandible_dir = case_dir / 'mandible'
    
    # Set output path
    if args.output is None:
        output_path = f'landmarks/{args.case}.json'
    else:
        output_path = args.output
    
    # Validate inputs
    if not dicom_dir.exists():
        print(f"❌ Error: DICOM directory not found: {dicom_dir}")
        sys.exit(1)
    
    if not mandible_dir.exists():
        print(f"❌ Error: Mandible directory not found: {mandible_dir}")
        sys.exit(1)
    
    # Find STL file
    stl_files = list(mandible_dir.glob('*.stl'))
    if len(stl_files) == 0:
        print(f"❌ Error: No STL file found in {mandible_dir}")
        sys.exit(1)
    stl_path = stl_files[0]
    
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    print(f"Case:           {args.case}")
    print(f"DICOM dir:      {dicom_dir}")
    print(f"STL file:       {stl_path.name}")
    print(f"HU threshold:   {args.hu_threshold}")
    print(f"Output:         {output_path}")
    print("="*60)
    
    # Load CT
    print("\n[1/4] Loading CT DICOM series...")
    ct_image = load_dicom_series(str(dicom_dir))
    
    print(f"\n[2/4] Extracting teeth surface (HU threshold: {args.hu_threshold})...")
    ct_mesh = extract_teeth_surface(ct_image, hu_threshold=args.hu_threshold)
    
    print("\n[3/4] Converting CT to point cloud...")
    ct_pcd = volume_to_pointcloud(ct_mesh, voxel_size=0.4)
    
    # Load STL
    print(f"\n[4/4] Loading and preparing STL...")
    stl_mesh = load_stl(str(stl_path))
    stl_pcd = prepare_pointcloud_for_icp(stl_mesh, target_points=8000)
    
    # Run landmark picker
    picker = LandmarkPicker(ct_pcd, stl_pcd, args.case)
    
    success = picker.run()
    
    if not success:
        print("\n❌ Landmark picking failed or cancelled.")
        sys.exit(1)
    
    # Save landmarks
    picker.save_landmarks(output_path)
    
    # Show preview if requested
    if args.preview:
        picker.compute_and_preview_transform()
    
    print("\n" + "="*60)
    print("LANDMARK PICKING COMPLETED")
    print("="*60)
    print(f"\nNext step: Run registration pipeline")
    print(f"  python main.py --case {args.case} --landmarks {output_path} --visualize")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
