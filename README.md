# Dental CT-STL Registration Pipeline

A high-performance toolkit for dental CBCT/CT image processing. This pipeline features automated DICOM-to-STL conversion using a hybrid dual-mask extraction strategy, performance-optimized parallel processing, and precision CT-STL point cloud registration.

---

## 🚀 Key Features

- **Hybrid Dual-Mask Strategy**: Sophisticated bone/tooth extraction that preserves delicate structures (e.g., maxillary sinus walls) while aggressively separating teeth.
- **Multiprocessing Parallelization**: High-speed batch processing utilizing multi-core CPU architecture (typically 5-10x faster).
- **Adaptive HU Thresholding**: Intelligent segmentation utilizing Otsu algorithms with spacing correction and safe clamping ranges.
- **Precision Registration**: Advanced three-stage ICP (Iterative Closest Point) alignment for sub-millimeter registration accuracy between IOS and CBCT.
- **Robust Component Filtering**: Relative volume filtering strategy that preserves disconnected anatomical parts (e.g., mandible and skull) while removing scattered noise.

---

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/dental-ct-stl-registration.git
cd dental-ct-stl-registration
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# Optional: for progress bars
pip install tqdm
```

---

## 📖 Module Reference

### 1. DICOM Batch Converter (`dicom_batch_to_ctl.py`)
Automates the conversion of CT volumes into meshes.

| Mode | Command | Description |
|------|---------|-------------|
| **Batch** | `--input-dir [PATH]` | Recursively scans for all DICOM series. |
| **Single** | `--single-dir [PATH]` | Processes one specific DICOM folder. |

**Example Usage:**
```bash
# Batch process a folder of cases
python dicom_batch_to_ctl.py --input-dir "./data/cases" --output-dir "./CT_CTL" --hu-threshold auto

# Parallel processing with specific worker count
python dicom_batch_to_ctl.py --input-dir "./data" --output-dir "./output" --workers 4
```

### 2. CT-STL Registration (`main.py`)
Aligns intraoral scans (STL) to the extracted CT bone surface.

```bash
python main.py --case 2023041102 --landmarks landmarks/2023041102.json --visualize
```

---

## 🔬 Technical Specifications

### Hybrid Dual-Mask Algorithm
To solve the trade-off between bone preservation and tooth separation:
1. **Base Mask**: Captures all bone (400-600 HU threshold). No erosion/opening applied to keep thin bone intact.
2. **Hard Mask**: Targets teeth (Base + 400 HU threshold). Aggressive Morphological Opening (0.5mm-0.8mm) to separate contact points.
3. **Fusion**: `(Base AND NOT Hard) OR Opened_Hard`.
4. **Post-Processing**: Morphological Closing (1.0mm) + Mesh-level Laplacian Smoothing.

### Adaptive Thresholding Logic
Calculates the optimal Hounsfield Unit (HU) for bone segmentation:
- **Phase 1**: Initial Otsu thresholding on ROI (300-2500 HU).
- **Phase 2**: Spacing correction (-150 * (effective_voxel_size - 0.3)).
- **Phase 3**: Clamp to [250-1000] HU.
- **Phase 4**: Apply user-defined offset (Default: +200 HU).

---

## 📈 Changelog

### v1.3.0
- **Parallelized Pipeline**: Implemented `ProcessPoolExecutor` for batch conversion.
- **Hybrid Extraction**: Added "Dual-Mask" logic to prevent bone erosion during tooth separation.
- **Worker Management**: New CLI options for `--workers` and `--no-parallel`.
- **UI/UX**: Added `tqdm` progress tracking and simplified logging output.

### v1.2.0
- Enhanced metal artifact reduction heuristics.
- Improved mandible protection in component filtering.
- Optimized memory usage with explicit garbage collection.


