# Dental CT-STL Registration Pipeline

A **stability-first, clinical-grade** registration system for aligning intraoral STL scans to CT coordinate systems for dental implant and surgical planning. Optimized for complex cases including heavy metal artifacts and varying scan resolutions.

---

## Overview

This pipeline addresses critical challenges in dental imaging workflows:

1. **Automated Surface Extraction**: Converts CT/CBCT DICOM volumes to watertight STL meshes
2. **Metal Artifact Reduction (MAR)**: Specialized algorithms for dental implants and fillings
3. **Precision Registration**: Aligns intraoral scan (IOS) STL data to CBCT coordinate systems

### Clinical Applications

- Surgical guide manufacturing
- Implant planning verification
- Digital orthodontic workflows
- CAD/CAM prosthetic design

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Adaptive HU Thresholding** | Otsu-based segmentation with P95 percentile hybrid logic |
| **Metal Artifact Compensation** | Automatic detection (P99 > 3000 HU) with threshold adjustment |
| **Morphological MAR** | Binary Opening operation to sever artifact connections |
| **Mesh Cluster Filtering** | Removes disconnected fragments < 100 triangles |
| **Three-Stage ICP** | Progressive refinement (5.0mm → 1.5mm → 0.8mm) |
| **Memory Management** | Garbage collection for large batch processing |
| **Batch Processing** | Recursive directory scanning with progress logging |

---

## System Requirements

### Minimum Configuration

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11, Ubuntu 20.04+, macOS 11+ |
| Python | 3.9 or higher |
| RAM | 16 GB (32 GB recommended for large volumes) |
| Storage | SSD recommended for DICOM I/O performance |

### Dependencies

```
open3d>=0.17.0
SimpleITK>=2.2.0
numpy>=1.21.0
scikit-image>=0.19.0
```

---

## Installation

### Option 1: Virtual Environment (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/dental-ct-stl-registration.git
cd dental-ct-stl-registration

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda Environment

```bash
conda create -n dental-reg python=3.10
conda activate dental-reg
pip install -r requirements.txt
```

---

## Quick Start

### 1. Batch DICOM to STL Conversion

```bash
python dicom_batch_to_ctl.py \
    --input-dir "./data/CT_cases" \
    --output-dir "./CT_CTL" \
    --hu-threshold auto
```

### 2. CT-STL Registration

```bash
python main.py \
    --case 2023042401 \
    --landmarks landmarks/2023042401.json \
    --interactive \
    --visualize
```

### 3. Point Cloud Visualization

```bash
python viewer.py \
    --dicom-dir "./data/CASE/cases/2023042401/DICOM" \
    --hu-threshold 1300 \
    --output-stl true
```

---

## Module Reference

### DICOM Batch Converter

**Script**: `dicom_batch_to_ctl.py`

Recursively scans directories for CT DICOM series and converts to STL format with automated preprocessing.

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir` | `str` | *required* | Root directory containing DICOM data |
| `--output-dir` | `str` | *required* | Destination for generated STL files |
| `--hu-threshold` | `str\|int` | `auto` | Segmentation threshold (`auto` or integer) |
| `--no-smooth` | `flag` | `False` | Disable anti-aliasing smoothing |

#### Output Naming Convention

```
{CaseID}_HU{threshold}.stl
```

Example: `2023042401_HU1285.stl`

#### Processing Pipeline

```
DICOM Load → Binary Threshold → Morphological Opening → 
CC Filtering → Anti-Alias → Marching Cubes → Mesh Cleanup → STL Export
```

---

### CT-STL Registration

**Script**: `main.py`

Performs rigid registration between CT-derived point clouds and intraoral scan STL meshes.

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--case` | `str` | *required* | Case identifier (subfolder name) |
| `--landmarks` | `str` | *required* | Path to landmarks JSON file |
| `--hu-threshold` | `int` | `1200` | CT segmentation threshold |
| `--interactive` | `flag` | `False` | Enable interactive landmark picker |
| `--visualize` | `flag` | `False` | Display before/after visualization |
| `--no-icp` | `flag` | `False` | Skip ICP refinement (landmarks only) |
| `--output-dir` | `str` | `processed` | Output directory |

#### Registration Stages

| Stage | Max Distance | Iterations | Purpose |
|-------|--------------|------------|---------|
| 1 - Expansion | 5.0 mm | 200 | Coarse alignment |
| 2 - Transitional | 1.5 mm | 100 | Intermediate refinement |
| 3 - Precision | 0.8 mm | 100 | Fine registration |

#### Quality Metrics

| Classification | RMSE Threshold | Inlier Ratio |
|----------------|----------------|--------------|
| **Reliable** ✓ | < 1.0 mm | > 60% |
| **Acceptable** | < 1.5 mm | > 45% |
| **Failed** ✗ | ≥ 1.5 mm | ≤ 45% |

---

### Point Cloud Viewer

**Script**: `viewer.py`

Interactive visualization tool for CT and STL point clouds with optional STL export.

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dicom-dir` | `str` | - | Path to DICOM directory |
| `--stl-path` | `str` | - | Path to STL file |
| `--mode` | `str` | `ct` | View mode: `ct`, `stl`, `both` |
| `--hu-threshold` | `str\|int` | `1200` | Threshold (`auto` or integer) |
| `--ct-voxel-size` | `float` | `0.4` | Point cloud sampling voxel size (mm) |
| `--output-stl` | `str` | - | Export path (use `true` for auto-naming) |
| `--no-smooth` | `flag` | `False` | Disable anti-aliasing |

---

## Technical Specifications

### Adaptive HU Threshold Algorithm

```
1. Extract ROI: 300 < HU < 2500 (bone/teeth range)
2. Apply Otsu thresholding on filtered data
3. Hybrid logic: 
   - If Otsu < 800: threshold = 0.4*Otsu + 0.6*P95
   - Else: threshold = Otsu
4. Spacing correction: -200 × (voxel_eff - 0.3)
5. Metal artifact: If P99 > 3000 → +150 HU
6. User offset: -450 HU (configurable)
7. Clamp to [500, 3000] HU
```

### Surface Extraction Pipeline

```
Binary Threshold (HU)
        ↓
Morphological Opening (0.5mm radius)
        ↓
Connected Component Filter (min 15 mm³)
        ↓
Anti-Alias Smoothing (RMS = 0.08 × voxel_eff)
        ↓
Marching Cubes (level = 0.0)
        ↓
Mesh Cleanup:
  - Remove duplicated vertices
  - Remove degenerate triangles
  - Cluster filtering (< 100 triangles)
```

### Coordinate System

| Source | System | Convention |
|--------|--------|------------|
| DICOM/SimpleITK | LPS | Left-Posterior-Superior |
| STL (typical) | RAS | Right-Anterior-Superior |

**Note**: This pipeline preserves native DICOM LPS coordinates.

---

## API Reference

### Core Functions

```python
# dicom_batch_to_ctl.py
calculate_adaptive_threshold(image: sitk.Image, logger) -> int
extract_surface_from_ct(image, hu_threshold, smooth, logger) -> o3d.TriangleMesh
convert_series_to_stl(series_path, dicom_names, output_dir, hu_threshold, smooth, logger) -> bool
batch_convert(input_dir, output_dir, hu_threshold, smooth, logger) -> Dict[str, int]

# main.py
compute_rigid_transform_svd(source_pts, target_pts) -> np.ndarray
refine_with_icp(source, target, initial_transform, skip_stage1) -> Tuple[np.ndarray, RegistrationResult]
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `No Series were found` | Empty subdirectory | Normal warning, can be ignored |
| Empty mesh output | Threshold too high | Lower `--hu-threshold` value |
| Memory error | Large volume | Increase system RAM or reduce batch size |
| KeyboardInterrupt | User abort | Normal termination signal |

### Metal Artifact Cases

For severe metal artifacts (dental implants, amalgam fillings):

```bash
# Use higher threshold manually
python viewer.py --dicom-dir "./case" --hu-threshold 1800

# Or adjust offset in code
final_threshold = final_threshold - 200  # More conservative
```

---

## Changelog

### v1.2.0 (2025-12-29)

**Added**
- Metal artifact reduction with morphological opening
- Mesh cluster filtering for noise removal
- Memory management with `gc.collect()`
- Smart output filename generation

**Changed**
- Increased threshold upper limit: 1800 → 3000 HU
- Metal artifact offset: -80 → +150 HU
- Morphology operation: Closing → Opening

### v1.1.0

- Three-stage ICP refinement
- Adaptive HU thresholding
- Physics-aware morphological operations

### v1.0.0

- Initial release

---

## Directory Structure

```
dental-ct-stl-registration/
├── main.py                     # Registration pipeline
├── viewer.py                   # Visualization tool
├── dicom_batch_to_ctl.py       # Batch converter
├── landmark_picker.py          # Standalone picker
├── requirements.txt
├── src/
│   ├── dicom_loader.py
│   ├── stl_loader.py
│   ├── registration.py
│   ├── visualizer.py
│   └── utils.py
├── data/                       # Input data
├── landmarks/                  # Landmark JSON files
├── processed/                  # Registration output
└── CT_CTL/                     # Batch conversion output
```

- ✅ Interactive Landmark Picking tool.

