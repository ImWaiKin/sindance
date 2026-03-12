"""
Data preprocessing module for fall risk prediction using MediaPipe keypoint data.

This module provides comprehensive preprocessing functionality for MediaPipe Holistic
pose landmark data used in fall risk prediction research. It handles the complete
pipeline from raw data files to cleaned, structured datasets ready for machine learning.

=== DATA FORMAT OVERVIEW ===

Raw Data Format:
- Input: Text files (.txt) containing timestamped keypoint sequences
- Each line: timestamp,kp1_idx,kp1_x,kp1_y,kp1_z,kp1_conf,kp1_vis,kp2_idx,kp2_x,kp2_y,...
- Each keypoint has 6 values: index (0-32), x, y, z coordinates, confidence, visibility
- Missing coordinates are marked as -1000
- Files follow naming convention: [eE]XX_YYYY-MM-DD_HH-MM.txt

MediaPipe Keypoints:
- 33 pose landmarks (indices 0-32) representing key body joints
- Coordinates: x,y (pixel locations), z (depth), confidence (0-1), visibility (0-1)
- Covers face/head (0-10), upper body (11-22), lower body (23-32)

Labels Format:
- CSV file with Subject_ID and Fall_Risk_Category columns
- Subject IDs match the [eE]XX pattern from filenames

=== PREPROCESSING PIPELINE ===

1. Data Loading:
   - Parse all .txt files in data directory
   - Extract subject IDs from filenames using regex
   - Convert raw text format to structured DataFrames

2. Data Cleaning:
   - Remove low confidence detections (< 0.5)
   - Filter out coordinate outliers using quantile or IQR methods
   - Handle missing values (-1000 → NaN)
   - Remove duplicate detections at same timestamp

3. Data Standardization:
   - Normalize timestamps to start from 0
   - Sort by timestamp for proper temporal ordering
   - Ensure consistent data structure across subjects

4. Data Validation:
   - Match subjects between keypoint data and labels
   - Report missing data and quality statistics
   - Filter to subjects with both data types

=== USAGE EXAMPLE ===

    from data_preprocessing import load_and_preprocess_data
    
    # Load and preprocess all data
    keypoint_data, labels_df = load_and_preprocess_data(
        data_dir='RawData_Dance',
        labels_file='fall_risk_categorization.csv'
    )
    
    # Access individual subject data
    subject_data = keypoint_data['E01']  # DataFrame for subject E01
    fall_risk = labels_df[labels_df['Subject_ID'] == 'E01']['Fall_Risk_Category'].iloc[0]

=== OUTPUT FORMAT ===

Processed keypoint data (per subject):
- Columns: timestamp, keypoint_idx, x, y, z, confidence, visibility
- Cleaned and filtered for reliable detections only
- Timestamps normalized to start from 0
- Ready for feature extraction and analysis

This preprocessing ensures high-quality, consistent data for fall risk prediction
model training and evaluation.
"""

import pandas as pd
import numpy as np
import os
import re
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm


class DataPreprocessor:
    """
    Main class for preprocessing MediaPipe keypoint data for fall risk prediction.
    
    This class encapsulates all the functionality needed to:
    1. Load raw keypoint data files from a directory
    2. Parse the complex keypoint format into structured DataFrames
    3. Clean and filter the data to remove noise and outliers
    4. Extract meaningful subject identifiers from filenames
    5. Synchronize keypoint data with fall risk labels
    6. Prepare data for machine learning pipeline
    """
    
    def __init__(self, data_dir: str = None, labels_file: str = None):
        """
        Initialize the data preprocessor with paths to data and labels.
        
        Args:
            data_dir: Directory containing raw keypoint data files (.txt format)
                     Each file represents one subject's movement recording
            labels_file: Path to the CSV file containing fall risk categorizations
                        Should have columns: Subject_ID, Fall_Risk_Category
        """
        # Store paths for data loading
        self.data_dir = data_dir
        self.labels_file = labels_file
        # Initialize MediaPipe keypoint mapping for reference
        self.keypoint_names = self._get_mediapipe_keypoint_names()
        
    def _get_mediapipe_keypoint_names(self) -> List[str]:
        """
        Get the ordered list of MediaPipe Holistic pose landmark names.
        
        MediaPipe Holistic provides 33 pose landmarks (indices 0-32) that represent
        key anatomical points on the human body. These landmarks are crucial for
        movement analysis and fall risk assessment.
        
        The landmarks cover:
        - Face/Head region (0-10): nose, eyes, ears, mouth
        - Upper body (11-22): shoulders, elbows, wrists, fingers
        - Lower body (23-32): hips, knees, ankles, feet
        
        Returns:
            List of 33 keypoint names in the order they appear in MediaPipe output
        """
        # MediaPipe Holistic landmarks (33 total: 0-32)
        # These correspond to specific body joints and anatomical markers
        return [
            # Head and face landmarks (0-10)
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            # Upper body landmarks (11-22) 
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            # Lower body landmarks (23-32)
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
    
    def _extract_subject_id(self, filename: str) -> str:
        """
        Extract standardized subject ID from data filename.
        
        Raw data files follow naming convention: [eE]XX_YYYY-MM-DD_HH-MM.txt
        where XX is the subject number (1-99+). This method extracts and standardizes
        the subject identifier to format "EXX" (e.g., "E01", "E58", "E33").
        
        Examples:
        - "e01_2025-03-03_11-40.txt" -> "E01"
        - "E33_2025-03-14_10-07.txt" -> "E33" 
        - "e58_2025-03-18_14-06.txt" -> "E58"
        
        Args:
            filename: Raw data filename to parse
            
        Returns:
            Standardized subject ID (e.g., "E01") or None if pattern not found
        """
        # Use regex to match patterns like e01, E01, e58, E33, etc.
        # [eE] matches either lowercase 'e' or uppercase 'E'
        # (\d+) captures one or more digits (subject number)
        match = re.search(r'[eE](\d+)', filename)
        if match:
            # Extract the numeric part and format as "EXX" with zero-padding
            # zfill(2) ensures single digits become "01", "02", etc.
            return f"E{match.group(1).zfill(2)}"
        return None
    
    def _parse_keypoint_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Parse a single keypoint data file into a structured DataFrame.
        
        Raw data format explanation:
        Each line in the file represents one frame of data with format:
        timestamp,kp1_idx,kp1_x,kp1_y,kp1_z,kp1_conf,kp1_vis,kp2_idx,kp2_x,kp2_y,kp2_z,kp2_conf,kp2_vis,...
        
        Where each keypoint has 6 values:
        - index: MediaPipe landmark index (0-32)
        - x, y: 2D pixel coordinates in the image
        - z: Depth/distance from camera (normalized)
        - confidence: MediaPipe's confidence in detection (0-1)
        - visibility: Whether the landmark is visible in frame (0-1)
        
        Special values:
        - "-1000" for x,y coordinates indicates missing/undetected keypoint
        - Low confidence values indicate unreliable detections
        
        Args:
            file_path: Path to the raw keypoint data file
            
        Returns:
            DataFrame with columns: timestamp, keypoint_idx, x, y, z, confidence, visibility
            Returns None if file parsing fails or contains no valid data
        """
        try:
            # Read all lines from the file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # List to store all parsed keypoint records
            data_rows = []
            
            # Process each line (frame) in the file
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Split the comma-separated values
                values = line.split(',')
                # Minimum requirement: timestamp + at least one complete keypoint (6 values)
                if len(values) < 7:  
                    continue
                
                # First value is always the timestamp (in seconds)
                timestamp = float(values[0])
                
                # Parse keypoints starting from index 1
                # Each keypoint has exactly 6 values: index, x, y, z, confidence, visibility
                keypoints = []
                for i in range(1, len(values), 6):  # Step by 6 to get each complete keypoint
                    # Ensure we have all 6 values for this keypoint
                    if i + 5 < len(values):
                        try:
                            # Parse individual keypoint data
                            kp_data = {
                                'index': int(values[i]),    # MediaPipe landmark index (0-32)
                                # Handle missing coordinates: -1000 -> NaN for proper handling
                                'x': float(values[i+1]) if values[i+1] != '-1000' else np.nan,
                                'y': float(values[i+2]) if values[i+2] != '-1000' else np.nan,
                                'z': float(values[i+3]) if values[i+3] != '-1000' else np.nan,   # Depth coordinate
                                'confidence': float(values[i+4]),  # Detection confidence (0-1)
                                'visibility': float(values[i+5])   # Visibility score (0-1)
                            }
                            keypoints.append(kp_data)
                        except (ValueError, IndexError):
                            # Skip malformed keypoint data but continue processing
                            continue
                
                # Convert keypoints to individual rows (one row per keypoint per timestamp)
                if keypoints:
                    for kp in keypoints:
                        data_rows.append({
                            'timestamp': timestamp,
                            'keypoint_idx': kp['index'],
                            'x': kp['x'],
                            'y': kp['y'], 
                            'z': kp['z'],
                            'confidence': kp['confidence'],
                            'visibility': kp['visibility']
                        })
            
            # Create DataFrame if we have valid data
            if data_rows:
                df = pd.DataFrame(data_rows)
                return df
            else:
                return None
                
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def load_labels(self) -> pd.DataFrame:
        """
        Load fall risk categorization labels from CSV file.
        
        The labels file should contain at least two columns:
        - Subject_ID: Identifier matching the keypoint data files (e.g., "E01", "E33")
        - Fall_Risk_Category: Classification (e.g., "Low", "Medium", "High")
        
        Returns:
            DataFrame containing subject IDs and their fall risk categories
        """
        labels_df = pd.read_csv(self.labels_file)
        return labels_df
    
    def load_all_keypoint_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and parse all keypoint data files from the data directory.
        
        This method scans the data directory for .txt files, extracts subject IDs
        from filenames, and parses each file's keypoint data into a DataFrame.
        Progress is shown via tqdm progress bar for large datasets.
        
        The process:
        1. Find all .txt files in the data directory
        2. Extract subject ID from each filename using regex pattern
        3. Parse keypoint data using _parse_keypoint_data()
        4. Store valid DataFrames in a dictionary keyed by subject ID
        
        Returns:
            Dictionary mapping subject IDs (str) to their keypoint DataFrames
            Only includes subjects with successfully parsed, non-empty data
        """
        keypoint_data = {}
        
        # Get all .txt files in the data directory
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        print(f"Found {len(data_files)} data files")
        
        # Process each file with progress tracking
        for filename in tqdm(data_files, desc="Loading keypoint data"):
            # Extract standardized subject ID from filename
            subject_id = self._extract_subject_id(filename)
            if subject_id is None:
                print(f"Could not extract subject ID from {filename}")
                continue
                
            # Parse the keypoint data file
            file_path = os.path.join(self.data_dir, filename)
            df = self._parse_keypoint_data(file_path)
            
            # Only store successfully parsed, non-empty DataFrames
            if df is not None and not df.empty:
                keypoint_data[subject_id] = df
            else:
                print(f"No valid data found in {filename}")
        
        print(f"Successfully loaded data for {len(keypoint_data)} subjects")
        return keypoint_data
    
    def load_keypoint_files(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Alternative method to load keypoint files from a specified directory.
        
        This is similar to load_all_keypoint_data() but allows specifying a different
        directory than the one set during initialization. Useful for loading test data
        or data from different sources.
        
        Args:
            data_dir: Directory containing keypoint files (.txt format)
            
        Returns:
            Dictionary mapping subject IDs to their parsed keypoint DataFrames
            
        Raises:
            FileNotFoundError: If the specified directory doesn't exist
        """
        keypoint_data = {}
        
        # Validate that the directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find all .txt files in the directory
        files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        print(f"Found {len(files)} keypoint files")
        
        # Process each file with progress tracking
        for filename in tqdm(files, desc="Loading keypoint files"):
            file_path = os.path.join(data_dir, filename)
            # Extract subject ID from filename
            subject_id = self._extract_subject_id(filename)
            
            if subject_id is None:
                print(f"Warning: Could not extract subject ID from {filename}")
                continue
            
            # Parse the keypoint data
            df = self._parse_keypoint_data(file_path)
            if df is not None and not df.empty:
                keypoint_data[subject_id] = df
                
        print(f"Successfully loaded data for {len(keypoint_data)} subjects")
        return keypoint_data
    
    def get_valid_subjects(self) -> List[str]:
        """
        Get list of subjects that have both keypoint data and fall risk labels.
        
        This method performs data validation by cross-referencing subjects in the
        keypoint data files with subjects in the labels CSV. Only subjects that
        appear in both datasets can be used for training and evaluation.
        
        The method also reports which subjects are missing from either dataset,
        which is helpful for data quality assessment and debugging.
        
        Returns:
            Sorted list of subject IDs that have both keypoint data and labels
            These subjects can be safely used for machine learning pipeline
        """
        # Load both datasets
        labels_df = self.load_labels()
        keypoint_data = self.load_all_keypoint_data()
        
        # Extract subject ID sets from both sources
        label_subjects = set(labels_df['Subject_ID'].values)
        keypoint_subjects = set(keypoint_data.keys())
        
        # Find intersection (subjects with both data types)
        valid_subjects = list(label_subjects.intersection(keypoint_subjects))
        
        # Report data availability statistics
        print(f"Found {len(valid_subjects)} subjects with both keypoint data and labels")
        print(f"Missing keypoint data: {label_subjects - keypoint_subjects}")
        print(f"Missing labels: {keypoint_subjects - label_subjects}")
        
        return sorted(valid_subjects)
    
    def preprocess_subject_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive preprocessing to keypoint data for a single subject.
        
        This method cleans and standardizes the raw keypoint data through several steps:
        1. Temporal ordering: Sort data by timestamp for proper sequence analysis
        2. Deduplication: Remove duplicate keypoint detections at same timestamp
        3. Quality filtering: Remove low-confidence keypoint detections
        4. Outlier removal: Remove extreme coordinate values that are likely errors
        5. Timestamp normalization: Reset timestamps to start from 0
        
        These preprocessing steps are crucial for reliable machine learning model training
        and ensure that the data represents actual human movement patterns.
        
        Args:
            df: Raw keypoint DataFrame for one subject with columns:
                [timestamp, keypoint_idx, x, y, z, confidence, visibility]
            
        Returns:
            Cleaned and preprocessed DataFrame with same structure but filtered data
        """
        # Step 1: Sort by timestamp to ensure proper temporal ordering
        # This is essential for time-series analysis and movement pattern recognition
        df = df.sort_values('timestamp').copy()
        
        # Step 2: Remove duplicate detections
        # Sometimes the same keypoint is detected multiple times at the same timestamp
        df = df.drop_duplicates(subset=['timestamp', 'keypoint_idx'])
        
        # Step 3: Filter out low confidence keypoints
        # MediaPipe confidence < 0.5 indicates unreliable detection
        # These can introduce noise and reduce model performance
        df = df[df['confidence'] > 0.5].copy()
        
        # Step 4: Remove coordinate outliers using quantile-based filtering
        # Extremely large coordinate values are usually detection errors
        for coord in ['x', 'y', 'z']:
            # Use 99th and 1st percentiles to identify outliers
            q99 = df[coord].quantile(0.99)
            q01 = df[coord].quantile(0.01)
            # Keep only values within the reasonable range
            df = df[(df[coord] <= q99) & (df[coord] >= q01)].copy()
        
        # Step 5: Normalize timestamps to start from 0
        # This standardizes sequences for consistent analysis across subjects
        if not df.empty:
            df.loc[:, 'timestamp'] = df['timestamp'] - df['timestamp'].min()
        
        return df
    
    def preprocess_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alternative preprocessing method using IQR-based outlier detection.
        
        This method uses a different outlier detection strategy compared to 
        preprocess_subject_data(). Instead of quantile-based filtering, it uses
        the Interquartile Range (IQR) method which is more robust for datasets
        with different distributions.
        
        Processing steps:
        1. Handle missing coordinates (already converted from -1000 to NaN)
        2. Filter low confidence detections
        3. Apply IQR-based outlier removal for x,y coordinates
        4. Normalize timestamps
        5. Sort by timestamp and keypoint index
        
        IQR method: Outliers are defined as values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR
        
        Args:
            df: Raw keypoint DataFrame for preprocessing
            
        Returns:
            Preprocessed DataFrame with outliers removed and timestamps normalized
        """
        # Early return for empty datasets
        if df.empty:
            return df
        
        # Missing coordinates (-1000 values) are already converted to NaN during parsing
        
        # Step 1: Remove low confidence keypoints (< 0.5 confidence threshold)
        df = df[df['confidence'] > 0.5].copy()
        
        # Step 2: Remove outliers using IQR (Interquartile Range) method
        # This is more robust than simple quantile filtering for skewed distributions
        for coord in ['x', 'y']:  # Only process x,y coordinates (z is depth)
            if coord in df.columns:
                # Calculate quartiles and IQR
                Q1 = df[coord].quantile(0.25)  # First quartile (25th percentile)
                Q3 = df[coord].quantile(0.75)  # Third quartile (75th percentile)
                IQR = Q3 - Q1                  # Interquartile range
                
                # Define outlier boundaries using 1.5 * IQR rule
                lower = Q1 - 1.5 * IQR         # Lower bound for outliers
                upper = Q3 + 1.5 * IQR         # Upper bound for outliers
                
                # Filter out outliers
                df = df[(df[coord] >= lower) & (df[coord] <= upper)]
        
        # Step 3: Normalize timestamps to start from 0
        # This ensures consistent time scales across different recording sessions
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'] - df['timestamp'].min()
        
        # Step 4: Sort by timestamp and keypoint index for consistent ordering
        # This is important for sequential analysis and feature extraction
        df = df.sort_values(['timestamp', 'keypoint_idx']).reset_index(drop=True)
        
        return df


def load_and_preprocess_data(data_dir: str = 'RawData_Dance', 
                           labels_file: str = 'fall_risk_categorization.csv') -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Main function to load and preprocess the complete dataset for fall risk prediction.
    
    This function orchestrates the entire data preprocessing pipeline:
    1. Initialize the DataPreprocessor with data paths
    2. Load fall risk labels from CSV file
    3. Load and parse all keypoint data files
    4. Apply preprocessing to each subject's data
    5. Filter labels to match available keypoint data
    6. Report final dataset statistics
    
    The function ensures data consistency by only including subjects that have
    both keypoint data and fall risk labels, making the dataset ready for
    machine learning model training and evaluation.
    
    Args:
        data_dir: Directory containing raw keypoint data files (.txt format)
                 Default: 'RawData_Dance' - expected to contain files like e01_*.txt
        labels_file: Path to CSV file with fall risk categorizations
                    Default: 'fall_risk_categorization.csv'
                    Expected columns: Subject_ID, Fall_Risk_Category
        
    Returns:
        Tuple containing:
        - keypoint_data_dict: Dictionary mapping subject IDs to preprocessed DataFrames
        - labels_dataframe: DataFrame with Subject_ID and Fall_Risk_Category columns
          filtered to only include subjects with valid keypoint data
          
    Example:
        keypoint_data, labels_df = load_and_preprocess_data()
        print(f"Loaded data for {len(keypoint_data)} subjects")
    """
    # Initialize preprocessor with data paths
    preprocessor = DataPreprocessor(data_dir, labels_file)
    
    # Step 1: Load fall risk categorization labels
    print("Loading fall risk labels...")
    labels_df = preprocessor.load_labels()
    
    # Step 2: Load and parse all raw keypoint data files
    print("Loading raw keypoint data...")
    raw_keypoint_data = preprocessor.load_all_keypoint_data()
    
    # Step 3: Apply preprocessing to each subject's data
    print("Applying preprocessing to each subject...")
    processed_keypoint_data = {}
    for subject_id, df in tqdm(raw_keypoint_data.items(), desc="Preprocessing subjects"):
        # Clean and filter the data (remove outliers, low confidence points, etc.)
        processed_df = preprocessor.preprocess_subject_data(df)
        # Only keep subjects with valid data after preprocessing
        if not processed_df.empty:
            processed_keypoint_data[subject_id] = processed_df
    
    # Step 4: Synchronize labels with available keypoint data
    # Only include subjects that have both keypoint data and labels
    valid_subjects = list(processed_keypoint_data.keys())
    labels_df = labels_df[labels_df['Subject_ID'].isin(valid_subjects)].copy()
    
    # Step 5: Report final dataset statistics
    print(f"\n=== Final Dataset Summary ===")
    print(f"Total subjects with valid data: {len(processed_keypoint_data)}")
    print(f"Fall risk distribution:")
    print(labels_df['Fall_Risk_Category'].value_counts())
    
    return processed_keypoint_data, labels_df


def extract_features_for_subject(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract basic features from a subject's keypoint data.
    
    This function extracts fundamental motion and postural features that serve
    as the foundation for advanced feature engineering. These basic features
    capture essential movement characteristics like velocity, spatial distribution,
    joint angles, and data quality metrics.
    
    Args:
        df: Preprocessed keypoint DataFrame for one subject with columns:
            ['timestamp', 'keypoint_idx', 'x', 'y', 'z', 'confidence', 'visibility']
            
    Returns:
        Dictionary of basic features:
        - mean_velocity: Average movement velocity across all keypoints
        - std_velocity: Standard deviation of velocity (movement variability)
        - foot_distance_std: Variability in distance between feet (stability)
        - knee_angles_mean: Average knee angle (mobility/flexibility)
        - core_mean_velocity: Average velocity of core/torso keypoints
        - z_range: Depth range of movement (3D space utilization)
        - x_mean: Average horizontal position (lateral preference)
        - mean_confidence: Average detection confidence (data quality)
    """
    import numpy as np
    from typing import Dict
    
    features = {}
    
    if df.empty:
        # Return zero features for empty data
        return {
            'mean_velocity': 0.0,
            'std_velocity': 0.0,
            'foot_distance_std': 0.0,
            'knee_angles_mean': 0.0,
            'core_mean_velocity': 0.0,
            'z_range': 0.0,
            'x_mean': 0.0,
            'mean_confidence': 0.0
        }
    
    # VELOCITY FEATURES
    # Calculate movement velocity for each keypoint over time
    velocities = []
    
    for kp_idx in df['keypoint_idx'].unique():
        kp_data = df[df['keypoint_idx'] == kp_idx].sort_values('timestamp')
        
        if len(kp_data) > 1:
            # Calculate frame-to-frame displacement
            coords = kp_data[['x', 'y', 'z']].values
            displacements = np.diff(coords, axis=0)
            times = kp_data['timestamp'].values
            time_diffs = np.diff(times)
            
            # Avoid division by zero
            time_diffs = np.where(time_diffs == 0, 1e-6, time_diffs)
            
            # Velocity magnitude for each frame
            velocity_mags = np.linalg.norm(displacements, axis=1) / time_diffs
            velocities.extend(velocity_mags)
    
    if velocities:
        features['mean_velocity'] = float(np.mean(velocities))
        features['std_velocity'] = float(np.std(velocities))
    else:
        features['mean_velocity'] = 0.0
        features['std_velocity'] = 0.0
    
    # FOOT DISTANCE FEATURES (stability)
    # Calculate distance between feet over time
    left_ankle = 27
    right_ankle = 28
    foot_distances = []
    
    for timestamp in df['timestamp'].unique():
        frame_data = df[df['timestamp'] == timestamp]
        left_foot = frame_data[frame_data['keypoint_idx'] == left_ankle]
        right_foot = frame_data[frame_data['keypoint_idx'] == right_ankle]
        
        if not left_foot.empty and not right_foot.empty:
            left_pos = left_foot[['x', 'y']].values[0]
            right_pos = right_foot[['x', 'y']].values[0]
            distance = np.linalg.norm(left_pos - right_pos)
            foot_distances.append(distance)
    
    if foot_distances:
        features['foot_distance_std'] = float(np.std(foot_distances))
    else:
        features['foot_distance_std'] = 0.0
    
    # KNEE ANGLE FEATURES
    # Calculate knee angles using hip-knee-ankle points
    knee_angles = []
    
    # Left knee angle (hip-knee-ankle)
    left_hip, left_knee, left_ankle = 23, 25, 27
    # Right knee angle (hip-knee-ankle)  
    right_hip, right_knee, right_ankle = 24, 26, 28
    
    for timestamp in df['timestamp'].unique():
        frame_data = df[df['timestamp'] == timestamp]
        
        # Calculate both knee angles
        for hip_idx, knee_idx, ankle_idx in [(left_hip, left_knee, left_ankle), 
                                           (right_hip, right_knee, right_ankle)]:
            hip_data = frame_data[frame_data['keypoint_idx'] == hip_idx]
            knee_data = frame_data[frame_data['keypoint_idx'] == knee_idx]
            ankle_data = frame_data[frame_data['keypoint_idx'] == ankle_idx]
            
            if not hip_data.empty and not knee_data.empty and not ankle_data.empty:
                hip_pos = hip_data[['x', 'y', 'z']].values[0]
                knee_pos = knee_data[['x', 'y', 'z']].values[0]
                ankle_pos = ankle_data[['x', 'y', 'z']].values[0]
                
                # Calculate angle at knee
                v1 = hip_pos - knee_pos
                v2 = ankle_pos - knee_pos
                
                # Handle zero vectors
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    knee_angles.append(angle)
    
    if knee_angles:
        features['knee_angles_mean'] = float(np.mean(knee_angles))
    else:
        features['knee_angles_mean'] = 0.0
    
    # CORE VELOCITY FEATURES
    # Focus on core/torso keypoints for representative body movement
    core_keypoints = [11, 12, 23, 24]  # shoulders and hips
    core_velocities = []
    
    for kp_idx in core_keypoints:
        kp_data = df[df['keypoint_idx'] == kp_idx].sort_values('timestamp')
        
        if len(kp_data) > 1:
            coords = kp_data[['x', 'y', 'z']].values
            displacements = np.diff(coords, axis=0)
            times = kp_data['timestamp'].values
            time_diffs = np.diff(times)
            time_diffs = np.where(time_diffs == 0, 1e-6, time_diffs)
            
            velocity_mags = np.linalg.norm(displacements, axis=1) / time_diffs
            core_velocities.extend(velocity_mags)
    
    if core_velocities:
        features['core_mean_velocity'] = float(np.mean(core_velocities))
    else:
        features['core_mean_velocity'] = 0.0
    
    # SPATIAL DISTRIBUTION FEATURES
    # Z-range (depth utilization)
    if 'z' in df.columns and not df['z'].isna().all():
        z_range = df['z'].max() - df['z'].min()
        features['z_range'] = float(z_range)
    else:
        features['z_range'] = 0.0
    
    # X-mean (lateral position preference)
    if 'x' in df.columns and not df['x'].isna().all():
        features['x_mean'] = float(df['x'].mean())
    else:
        features['x_mean'] = 0.0
    
    # DATA QUALITY FEATURES
    # Average confidence score
    if 'confidence' in df.columns and not df['confidence'].isna().all():
        features['mean_confidence'] = float(df['confidence'].mean())
    else:
        features['mean_confidence'] = 0.0
    
    return features


if __name__ == "__main__":
    """
    Test and demonstration section for the data preprocessing module.
    
    When this file is run directly (not imported), it executes a complete
    preprocessing pipeline test that:
    1. Loads and preprocesses all available data
    2. Displays overall dataset statistics
    3. Shows detailed statistics for the first 3 subjects as examples
    
    This is useful for:
    - Testing the preprocessing pipeline
    - Debugging data loading issues
    - Understanding data characteristics
    - Validating preprocessing results
    """
    print("=== Testing Data Preprocessing Pipeline ===")
    print("Loading and preprocessing all data...")
    
    # Execute the complete preprocessing pipeline
    keypoint_data, labels_df = load_and_preprocess_data()
    
    print("\n=== Sample Subject Statistics ===")
    print("Showing detailed statistics for first 3 subjects:")
    
    # Display detailed statistics for the first 3 subjects as examples
    for subject_id, df in list(keypoint_data.items())[:3]:
        print(f"\nSubject {subject_id}:")
        print(f"  Total data points: {len(df)}")
        print(f"  Recording duration: {df['timestamp'].max():.2f} seconds")
        print(f"  Unique keypoints detected: {df['keypoint_idx'].nunique()}/33")
        print(f"  Mean detection confidence: {df['confidence'].mean():.3f}")
        print(f"  Mean visibility score: {df['visibility'].mean():.3f}")
        
        # Show coordinate ranges to understand movement space
        print(f"  X coordinate range: [{df['x'].min():.1f}, {df['x'].max():.1f}]")
        print(f"  Y coordinate range: [{df['y'].min():.1f}, {df['y'].max():.1f}]")
        print(f"  Z coordinate range: [{df['z'].min():.3f}, {df['z'].max():.3f}]")
    
    print(f"\n=== Data Quality Summary ===")
    print(f"Successfully processed {len(keypoint_data)} subjects")
    print("Data is ready for feature extraction and model training!")
