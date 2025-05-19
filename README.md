# TFCE Moments Eyetracking Toolbox

A collection of MATLAB scripts for advanced spatial statistical analysis of eye-tracking data.

## Overview

This toolbox provides three complementary methods for analyzing eye-tracking data:

1. Bilateral peak detection and data extraction
2. Two-dimensional higher-order statistical moments analysis  
3. Threshold-free cluster enhancement (TFCE) analysis

## Prerequisites

### Required Software
- MATLAB R2017b or later
- CoSMoMVPA toolbox (for TFCE analysis)
- **GazeAnalysisGambling package** (prerequisite for data preprocessing)

### Required External Functions
These scripts require external functions from MATLAB File Exchange:

* readfromexcel:
  - File ID: 4415

* xlswrite:
  - File ID: 7881

### Input Data Requirements
- **Gaze ratio Excel files from `calculate_gaze_ratios_v1.m`** (GazeAnalysisGambling package)
- Two-group experimental design
- Data should be preprocessed using GazeAnalysisGambling scripts up to step 2

## Directory Structure

    tfce_moments_eyetracking_toolbox/
    ├── Input_Group1/            # Group 1 gaze ratio files
    ├── Input_Group2/            # Group 2 gaze ratio files
    ├── detect_bilateral_peaks_and_extract_v1.m
    ├── two_sample_TDM_v1.m
    └── two_sample_TFCE_v1.m

Note: The `.gitkeep` files in empty directories are used to maintain the directory structure in GitHub. These files can be safely kept or deleted after downloading the repository - they do not affect the scripts' functionality.

## Usage Workflow

### Step 1: Data Preparation
1. Complete preprocessing with GazeAnalysisGambling package:
   - Run `calculate_gaze_ratios_v1.m`
2. Place the output gaze ratio Excel files in `Input_Group1` and `Input_Group2` folders

### Step 2: Peak Detection and Extraction
```matlab
% Run peak detection script
detect_bilateral_peaks_and_extract_v1
```

This script:
- Detects peak gaze positions in left and right hemifields
- Extracts 201×201 pixel regions around peaks
- Saves peak coordinates and extracted data

### Step 3: Distribution Analysis
```matlab
% Run TDM analysis
two_sample_TDM_v1
```

This script:
- Calculates 2D skewness and kurtosis (Mardia's measures)
- Performs statistical comparisons between groups
- Generates visualization outputs

### Step 4: TFCE Statistical Analysis
```matlab
% Run TFCE analysis
two_sample_TFCE_v1
```

This script:
- Performs cluster-based permutation testing
- Identifies significant spatial differences
- Generates comprehensive statistical maps

## Script Descriptions

### 1. `detect_bilateral_peaks_and_extract_v1.m`

**Purpose**: Identifies peak gaze positions and extracts surrounding data.

**Key Parameters**:
- `Resolution_W`: Screen width (default: 1024)
- `Range`: Extraction area radius (default: 100)

**Input**: Gaze ratio Excel files from `calculate_gaze_ratios_v1.m`

**Output**:
- `detected_peaks_group1.xlsx` and `detected_peaks_group2.xlsx`
- Individual peak area data files

### 2. `two_sample_TDM_v1.m`

**Purpose**: Analyzes 2D distribution properties using higher-order moments.

**Key Parameters**:
- `Resolution_X`, `Resolution_Y`: Analysis resolution (default: 201×201)
- `sigma`: Gaussian smoothing (default: 10)
- `alpha`: Significance level (default: 0.05)

**Analysis Methods**:
- 2D skewness (distribution asymmetry)
- 2D kurtosis (distribution peakedness)
- Independent samples t-tests
- Bonferroni correction

### 3. `two_sample_TFCE_v1.m`

**Purpose**: Performs threshold-free cluster enhancement analysis.

**Key Parameters**:
- `E`: TFCE extent parameter (default: 0.5)
- `H`: TFCE height parameter (default: 2)
- `niter`: Permutations (default: 1000)
- `connect`: Pixel connectivity (default: 8)

**Analysis Methods**:
- CoSMoMVPA-based permutation testing
- Cluster identification
- Statistical map generation

## Output Interpretation

### TDM Results
- **Skewness**: Measures distribution asymmetry
  - Positive: Tail on right/upper side
  - Negative: Tail on left/lower side
- **Kurtosis**: Measures distribution peakedness
  - Positive: Heavy tails, peaked center
  - Negative: Light tails, flat center

### TFCE Results
- **Z-statistics maps**: Spatial group differences
- **Significance clusters**: Areas with reliable differences
- **Cluster statistics**: Size, peak values, locations

## Important Notes

- This toolbox requires preprocessed data from GazeAnalysisGambling
- Ensure CoSMoMVPA is properly installed and added to MATLAB path
- Analysis parameters should be adjusted based on your specific data
- Screen resolution and sampling rate may require parameter modifications

## Input File Format

Input files should be Excel files with columns:
- Column 1: X coordinates
- Column 2: Y coordinates  
- Column 3-6: Phase-specific gaze ratios (Decision/Feedback, Left/Right)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolbox in your research, please cite our work:

(Under Review)

## Acknowledgments

This toolbox extends the functionality of GazeAnalysisGambling by providing advanced spatial statistical methods for eye-tracking data analysis.

## Support

For technical issues, please refer to the documentation within individual scripts.
