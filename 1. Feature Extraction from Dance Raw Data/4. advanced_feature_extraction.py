"""
Advanced feature extraction module for fall risk prediction.

This module implements sophisticated feature engineering techniques to improve model accuracy
for fall risk assessment using pose estimation data. The features are organized into several
categories, each capturing different aspects of human movement and postural control:

FEATURE CATEGORIES:
==================

1. TEMPORAL SEQUENCE FEATURES:
   - com_autocorr_1: First-order autocorrelation of center of mass movement
   - com_trend: Linear trend in center of mass displacement over time
   - movement_reversals: Frequency of movement direction changes (normalized)
   - movement_regularity: Rhythmic consistency of movement patterns
   - stability_trend: How stability changes over time
   - stability_variance: Consistency of stability over time
   - stability_min: Worst stability moment during recording

2. FREQUENCY DOMAIN FEATURES:
   - dominant_frequency: Main oscillatory frequency in movement
   - spectral_centroid: Weighted average frequency ("center of mass" of spectrum)
   - spectral_spread: Frequency bandwidth around spectral centroid
   - spectral_energy: Total energy across all frequencies
   - freq_power_ratio: Ratio of low-frequency to high-frequency power

3. BODY SYMMETRY FEATURES:
   - left_arm_asymmetry: Movement asymmetry between left and right arms
   - left_leg_asymmetry: Movement asymmetry between left and right legs

4. DYNAMIC STABILITY FEATURES:
   - base_support_mean: Average base of support (distance between ankles)
   - base_support_std: Variability in base of support
   - base_support_min: Minimum base of support observed
   - cop_velocity: Average velocity of center of pressure movement
   - cop_acceleration: Average acceleration of center of pressure
   - cop_path_length: Total path length traveled by center of pressure

5. COORDINATION FEATURES:
   - [limb1]_[limb2]_coordination: Cross-correlation between limb movements
   - [limb1]_[limb2]_phase_coupling: Phase synchronization between limbs

6. POSTURAL SWAY FEATURES:
   - sway_area: Area covered by center of gravity trajectory
   - sway_velocity: Average velocity of postural sway
   - sway_acceleration: Average acceleration of postural sway
   - ml_sway: Medio-lateral (side-to-side) sway variability
   - ap_sway: Anterior-posterior (forward-backward) sway variability

7. JOINT COORDINATION FEATURES:
   - [joint1]_[joint2]_coord: Coordination between adjacent joints in kinetic chain

8. INTERACTION FEATURES:
   - velocity_cv: Coefficient of variation for velocity
   - balance_coordination: Interaction between balance and joint coordination
   - interaction_[feature1]_x_[feature2]: Polynomial interactions between features

CLINICAL RELEVANCE:
==================
- Higher asymmetry values indicate potential neurological or musculoskeletal issues
- Excessive sway metrics suggest poor balance control and increased fall risk
- Poor coordination values indicate movement dysfunction
- High frequency content may indicate tremors or instability
- Irregular temporal patterns suggest loss of motor control

FALL RISK INTERPRETATION:
========================
- LOW RISK: High regularity, low asymmetry, controlled sway, good coordination
- HIGH RISK: Irregular patterns, high asymmetry, excessive sway, poor coordination
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats, signal
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureExtractor:
    """Advanced feature extraction for improved fall risk prediction."""
    
    def __init__(self):
        """Initialize the advanced feature extractor."""
        self.body_parts = {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'torso': [11, 12, 23, 24],
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22],
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32],
            'core': [11, 12, 23, 24],
            'extremities': [15, 16, 27, 28]  # hands and feet
        }
        
        # Key joints for biomechanical analysis
        self.key_joints = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
    def _calculate_temporal_sequences(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract temporal sequence and pattern features.
        
        These features capture how movement patterns evolve over time and assess
        the temporal consistency and predictability of motion patterns.
        """
        features = {}
        
        # Calculate movement sequences for center of mass
        com_data = self._calculate_center_of_mass_series(df)
        if len(com_data) > 5:
            # TEMPORAL AUTOCORRELATION FEATURES
            # com_autocorr_1: First-order autocorrelation of center of mass displacement
            # Measures how predictable the movement is from one time step to the next
            # Higher values indicate more regular, predictable movement patterns
            # Lower values suggest more erratic, unpredictable movement (potential fall risk)
            features['com_autocorr_1'] = np.corrcoef(com_data[:-1], com_data[1:])[0, 1] if len(com_data) > 1 else 0
            
            # com_trend: Linear trend in center of mass displacement over time
            # Derived using linear regression slope of COM displacement vs. time
            # Positive trend indicates overall movement away from starting position
            # Negative trend indicates movement toward starting position
            # Large absolute values may indicate loss of postural control
            features['com_trend'] = stats.linregress(range(len(com_data)), com_data)[0]
            
            # MOVEMENT DIRECTION CHANGES
            # movement_reversals: Normalized count of movement direction changes
            # Calculated by detecting sign changes in the derivative of COM displacement
            # High values indicate frequent direction changes (shakiness, instability)
            # Low values suggest smoother, more controlled movement
            reversals = np.sum(np.diff(np.sign(np.diff(com_data))) != 0)
            features['movement_reversals'] = reversals / len(com_data)
            
            # RHYTHMICITY AND REGULARITY
            # movement_regularity: Quantifies the rhythmic consistency of movement
            # Uses autocorrelation analysis to detect periodic patterns
            # Higher values indicate more regular, rhythmic movement patterns
            # Lower values suggest irregular, uncoordinated movement
            features['movement_regularity'] = self._calculate_regularity(com_data)
        
        # TEMPORAL STABILITY ANALYSIS
        # Analyzes how stability changes over time windows
        stability_series = self._calculate_stability_over_time(df)
        if len(stability_series) > 5:
            # stability_trend: How stability changes over time
            # Positive trend: stability improving over time
            # Negative trend: stability deteriorating over time (fall risk indicator)
            features['stability_trend'] = stats.linregress(range(len(stability_series)), stability_series)[0]
            
            # stability_variance: Consistency of stability over time
            # High variance indicates inconsistent stability (potential fall risk)
            # Low variance suggests consistent postural control
            features['stability_variance'] = np.var(stability_series)
            
            # stability_min: Worst stability moment during the recording
            # Lower values indicate periods of very poor stability
            # Critical for identifying fall risk episodes
            features['stability_min'] = np.min(stability_series)
        
        return features
    
    def _calculate_frequency_domain_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT analysis.
        
        These features analyze the frequency content of movement patterns to identify
        tremors, oscillations, and rhythmic patterns that may indicate fall risk.
        """
        features = {}
        
        # Calculate velocity magnitude over time
        velocity_series = self._get_velocity_time_series(df)
        
        if len(velocity_series) > 10:
            # Apply FFT (Fast Fourier Transform) to convert time-domain to frequency-domain
            fft_values = fft(velocity_series)
            freqs = fftfreq(len(velocity_series))
            
            # POWER SPECTRAL DENSITY ANALYSIS
            # Use only positive frequency components for physically meaningful features
            psd = np.abs(fft_values) ** 2
            pos_mask = freqs > 0
            pos_freqs = freqs[pos_mask]
            pos_psd = psd[pos_mask]
            
            # FREQUENCY CONTENT FEATURES
            # dominant_frequency: The frequency with the highest power
            # Identifies the main oscillatory component in movement
            features['dominant_frequency'] = pos_freqs[np.argmax(pos_psd)] if len(pos_psd) > 0 else 0
            
            # spectral_centroid: Weighted average of frequencies (frequency "center of mass")
            pos_psd_sum = np.sum(pos_psd)
            features['spectral_centroid'] = np.sum(pos_freqs * pos_psd) / pos_psd_sum if pos_psd_sum > 0 else 0
            
            # spectral_spread: Spread of frequencies around the spectral centroid
            features['spectral_spread'] = np.sqrt(np.sum(((pos_freqs - features['spectral_centroid']) ** 2) * pos_psd) / pos_psd_sum) if pos_psd_sum > 0 else 0
            
            # spectral_energy: Total energy across positive frequencies
            features['spectral_energy'] = pos_psd_sum
            
            # FREQUENCY BAND ANALYSIS
            # freq_power_ratio: Ratio of low-frequency to high-frequency power
            # Uses positive frequencies only for correct band separation
            low_freq_power = np.sum(pos_psd[pos_freqs < 0.1])
            high_freq_power = np.sum(pos_psd[pos_freqs >= 0.1])
            features['freq_power_ratio'] = low_freq_power / (high_freq_power + 1e-6)
        
        return features
    
    def _calculate_biomechanical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract advanced biomechanical and stability features.
        
        These features analyze body mechanics, postural control, and movement coordination
        to assess fall risk from a biomechanical perspective.
        """
        features = {}
        
        # BODY SYMMETRY ANALYSIS
        # Analyzes left-right body symmetry during movement
        # Asymmetric movement patterns may indicate neurological issues or injury
        symmetry_features = self._calculate_body_symmetry(df)
        features.update(symmetry_features)
        
        # DYNAMIC STABILITY INDICATORS
        # Measures how well the body maintains stability during movement
        # Includes base of support analysis and center of pressure metrics
        stability_features = self._calculate_dynamic_stability(df)
        features.update(stability_features)
        
        # COORDINATION METRICS
        # Analyzes inter-limb coordination and movement synchronization
        # Poor coordination is a strong predictor of fall risk
        coordination_features = self._calculate_coordination_metrics(df)
        features.update(coordination_features)
        
        # POSTURAL SWAY ANALYSIS
        # Measures body sway patterns and postural control
        # Excessive sway indicates poor balance and increased fall risk
        sway_features = self._calculate_postural_sway(df)
        features.update(sway_features)
        
        # MULTI-JOINT COORDINATION
        # Analyzes coordination between different joints
        # Joint coordination is crucial for smooth, stable movement
        joint_features = self._calculate_joint_coordination(df)
        features.update(joint_features)
        
        return features
    
    def _calculate_interaction_features(self, base_features: Dict[str, float]) -> Dict[str, float]:
        """
        Create interaction features from base features.
        
        Interaction features capture non-linear relationships between basic features
        that may be more predictive of fall risk than individual features alone.
        """
        features = {}
        
        # Select important features for interactions
        important_features = [
            'mean_velocity', 'std_velocity', 'foot_distance_std', 'knee_angles_mean',
            'core_mean_velocity', 'z_range', 'x_mean', 'mean_confidence'
        ]
        
        # POLYNOMIAL INTERACTION FEATURES
        # Create second-degree polynomial interactions between features
        # These capture non-linear relationships that may be missed by linear models
        feature_values = []
        feature_names = []
        
        for feat in important_features:
            if feat in base_features:
                feature_values.append(base_features[feat])
                feature_names.append(feat)
        
        if len(feature_values) >= 2:
            feature_array = np.array(feature_values).reshape(1, -1)
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_features = poly.fit_transform(feature_array)[0]
            poly_names = poly.get_feature_names_out(feature_names)
            
            # Add polynomial interaction features
            # These represent products of pairs of basic features
            for i, name in enumerate(poly_names):
                if ' ' in name:  # Only interaction terms (not squared terms)
                    clean_name = f"interaction_{name.replace(' ', '_').replace('*', 'x')}"
                    features[clean_name] = poly_features[i]
        
        # MANUALLY CRAFTED IMPORTANT INTERACTIONS
        # velocity_cv: Coefficient of variation for velocity
        # Measures relative variability in movement speed
        # High CV indicates inconsistent movement patterns (potential fall risk)
        # Low CV suggests more controlled, consistent movement
        if 'mean_velocity' in base_features and 'std_velocity' in base_features:
            features['velocity_cv'] = base_features['std_velocity'] / (base_features['mean_velocity'] + 1e-6)
        
        # balance_coordination: Interaction between balance and joint coordination
        # Combines foot stability (foot_distance_std) with knee control (knee_angles_mean)
        # Higher values may indicate compensation strategies or poor coordination
        # Lower values suggest better integrated balance and coordination
        if 'foot_distance_std' in base_features and 'knee_angles_mean' in base_features:
            features['balance_coordination'] = base_features['foot_distance_std'] * base_features['knee_angles_mean']
        
        return features
    
    def _calculate_center_of_mass_series(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate center of mass displacement over time.
        
        The center of mass represents the average position of all body keypoints,
        weighted by their confidence scores. Changes in COM position over time
        indicate overall body movement patterns and postural control.
        """
        # Vectorised weighted-average COM per timestamp (no per-row loop)
        valid = df.dropna(subset=['x', 'y']).copy()
        if valid.empty:
            return np.array([])
        valid['wx'] = valid['x'] * valid['confidence']
        valid['wy'] = valid['y'] * valid['confidence']
        g = valid.groupby('timestamp')
        sum_wx = g['wx'].sum()
        sum_wy = g['wy'].sum()
        sum_w  = g['confidence'].sum()
        com_x = (sum_wx / sum_w).sort_index()
        com_y = (sum_wy / sum_w).sort_index()
        return np.sqrt(com_x.values ** 2 + com_y.values ** 2)
    
    def _calculate_regularity(self, series: np.ndarray) -> float:
        """
        Calculate movement regularity using autocorrelation.
        
        Regularity measures how predictable and rhythmic movement patterns are.
        Higher regularity indicates more controlled, consistent movement.
        Lower regularity suggests erratic, unpredictable movement patterns.
        """
        if len(series) < 5:
            return 0
        
        # DETRENDING
        # Remove linear trends to focus on oscillatory patterns
        detrended = signal.detrend(series)
        
        # AUTOCORRELATION ANALYSIS
        # Calculate autocorrelation to find repetitive patterns
        autocorr = np.correlate(detrended, detrended, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Take positive lags only
        if autocorr[0] == 0:  # Constant signal after detrending
            return 0
        autocorr = autocorr / autocorr[0]  # Normalize by zero-lag value
        
        # PEAK DETECTION
        # Find peaks in autocorrelation indicating periodic patterns
        # Higher peaks suggest more regular, rhythmic movement
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
        
        # Regularity score: normalized number of significant peaks
        return len(peaks) / len(autocorr) if len(autocorr) > 0 else 0
    
    def _calculate_stability_over_time(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate stability metric over time windows.
        
        Analyzes how postural stability changes over the duration of movement.
        This temporal analysis can reveal patterns of fatigue, adaptation,
        or deteriorating balance control over time.
        """
        stability_series = []
        
        timestamps = sorted(df['timestamp'].unique())
        window_size = max(5, len(timestamps) // 10)  # Adaptive window size
        
        # SLIDING WINDOW ANALYSIS
        # Calculate stability for overlapping time windows
        for i in range(0, len(timestamps) - window_size + 1, window_size // 2):
            window_timestamps = timestamps[i:i + window_size]
            window_data = df[df['timestamp'].isin(window_timestamps)]
            
            # STABILITY CALCULATION
            # Stability as inverse of movement variance
            # Higher variance = lower stability
            if len(window_data) > 0:
                movement_var = (window_data['x'].var() + window_data['y'].var()) / 2
                stability = 1 / (1 + movement_var)  # Inverse relationship
                stability_series.append(stability)
        
        return np.array(stability_series)
    
    def _get_velocity_time_series(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get velocity magnitude time series for frequency analysis.
        
        Computes a single, temporally ordered velocity series by averaging core
        keypoint positions per timestamp and calculating time-normalised velocity.
        This produces a proper time series suitable for FFT analysis.
        """
        # Average position of core keypoints per timestamp
        core_keypoints = self.body_parts['core']
        core_data = df[df['keypoint_idx'].isin(core_keypoints)]
        if core_data.empty:
            return np.array([0])
        
        mean_pos = core_data.groupby('timestamp', sort=True)[['x', 'y', 'z']].mean().dropna()
        if len(mean_pos) < 3:
            return np.array([0])
        
        positions = mean_pos.values
        timestamps = mean_pos.index.values.astype(float)
        
        # Frame-to-frame displacement normalised by time
        displacements = np.diff(positions, axis=0)
        time_diffs = np.diff(timestamps)
        time_diffs = np.where(time_diffs == 0, 1e-6, time_diffs)
        
        velocity_mags = np.linalg.norm(displacements, axis=1) / time_diffs
        return velocity_mags
    
    def _calculate_body_symmetry(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate body symmetry features.
        
        Body symmetry analysis compares left and right sides of the body to detect
        asymmetric movement patterns that may indicate neurological issues, injury,
        or compensatory movement strategies.
        """
        features = {}
        
        # BILATERAL SYMMETRY ANALYSIS
        # Compare movement patterns between left and right body parts
        symmetry_pairs = [
            ('left_arm', 'right_arm'),
            ('left_leg', 'right_leg')
        ]
        
        for left_part, right_part in symmetry_pairs:
            left_keypoints = self.body_parts[left_part]
            right_keypoints = self.body_parts[right_part]
            
            left_data = df[df['keypoint_idx'].isin(left_keypoints)]
            right_data = df[df['keypoint_idx'].isin(right_keypoints)]
            
            if not left_data.empty and not right_data.empty:
                # MOVEMENT ASYMMETRY CALCULATION
                # Calculate standard deviation of movement for each side
                # Higher std indicates more movement/variability
                left_movement = left_data[['x', 'y', 'z']].std().mean()
                right_movement = right_data[['x', 'y', 'z']].std().mean()
                
                # asymmetry: Normalized difference between left and right movement
                # Values range from 0 (perfect symmetry) to 1 (complete asymmetry)
                # Higher values indicate greater asymmetry (potential fall risk)
                # Calculated as |left - right| / (left + right + epsilon)
                asymmetry = abs(left_movement - right_movement) / (left_movement + right_movement + 1e-6)
                features[f'{left_part}_asymmetry'] = asymmetry
        
        return features
    
    def _calculate_dynamic_stability(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate dynamic stability indicators.
        
        Dynamic stability measures how well the body maintains balance during movement.
        These features are crucial for fall risk assessment as they directly relate
        to the body's ability to maintain upright posture.
        """
        features = {}
        
        # BASE OF SUPPORT ANALYSIS
        # Base of support is the area between the feet - fundamental for stability
        ankle_keypoints = [27, 28]  # left and right ankles
        ankle_data = df[df['keypoint_idx'].isin(ankle_keypoints)]
        
        if not ankle_data.empty:
            # Pivot once to get left/right ankle per timestamp
            try:
                ankle_wide = ankle_data.pivot_table(
                    index='timestamp', columns='keypoint_idx', values=['x', 'y'], aggfunc='first'
                )
                base_areas = []
                for _, row in ankle_wide.iterrows():
                    try:
                        lx, ly = row[('x', 27)], row[('y', 27)]
                        rx, ry = row[('x', 28)], row[('y', 28)]
                        if not any(pd.isna(v) for v in [lx, ly, rx, ry]):
                            base_areas.append(np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2))
                    except (KeyError, TypeError):
                        continue
            except Exception:
                base_areas = []
            
            if base_areas:
                features['base_support_mean'] = np.mean(base_areas)
                features['base_support_std'] = np.std(base_areas)
                features['base_support_min'] = np.min(base_areas)
        
        # CENTER OF PRESSURE ANALYSIS
        # Analyzes the movement of the center of pressure under the feet
        # Important for understanding weight distribution and balance control
        features.update(self._calculate_center_of_pressure_features(df))
        
        return features
    
    def _calculate_coordination_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate inter-limb coordination metrics.
        
        Coordination between limbs is essential for smooth, stable movement.
        Poor coordination is a strong predictor of fall risk, especially in
        complex movements or challenging environments.
        """
        features = {}
        
        # CROSS-CORRELATION ANALYSIS BETWEEN LIMBS
        # Measures how synchronized different limb movements are
        limb_pairs = [
            ('left_arm', 'right_arm'),    # Bilateral arm coordination
            ('left_leg', 'right_leg'),    # Bilateral leg coordination  
            ('left_arm', 'left_leg'),     # Ipsilateral coordination
            ('right_arm', 'right_leg')    # Ipsilateral coordination
        ]
        
        for limb1, limb2 in limb_pairs:
            # coordination values: Cross-correlation between limb movement patterns
            # Range from -1 (perfectly anti-correlated) to 1 (perfectly correlated)
            # Values near 0 indicate poor coordination
            # High positive values indicate good synchronization
            # High negative values may indicate alternating patterns (e.g., walking)
            correlation = self._calculate_limb_correlation(df, limb1, limb2)
            features[f'{limb1}_{limb2}_coordination'] = correlation
        
        # PHASE COUPLING ANALYSIS
        # Analyzes the phase relationships between limb movements
        # Important for understanding rhythmic coordination patterns
        features.update(self._calculate_phase_coupling(df))
        
        return features
    
    def _calculate_postural_sway(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate postural sway metrics.
        
        Postural sway measures the movement of the body's center of gravity.
        Excessive sway indicates poor balance control and increased fall risk.
        These metrics are fundamental in clinical balance assessment.
        """
        features = {}
        
        # CENTER OF GRAVITY APPROXIMATION
        # Use hip midpoint as proxy for center of gravity
        # Hips represent the body's center of mass better than other landmarks
        hip_data = df[df['keypoint_idx'].isin([23, 24])]  # left and right hips
        
        if not hip_data.empty:
            # Vectorised: mean x/y of both hips per timestamp
            hip_mean = hip_data.groupby('timestamp', sort=True)[['x', 'y']].mean().dropna()
            hip_centers = hip_mean.values
            
            if len(hip_centers) > 5:
                hip_trajectory = np.array(hip_centers)
                
                # SWAY AREA AND MOVEMENT METRICS
                # sway_area: Area covered by the center of gravity trajectory
                # Larger areas indicate more postural instability
                # Calculated using convex hull of trajectory points
                features['sway_area'] = self._calculate_sway_area(hip_trajectory)
                
                # sway_velocity: Average velocity of center of gravity movement
                # Higher velocities indicate more rapid postural adjustments
                # May suggest poor balance control or compensation strategies
                features['sway_velocity'] = self._calculate_sway_velocity(hip_trajectory)
                
                # sway_acceleration: Average acceleration of sway movement
                # High accelerations indicate sudden, jerky movements
                # Often associated with balance corrections and fall risk
                features['sway_acceleration'] = self._calculate_sway_acceleration(hip_trajectory)
                
                # DIRECTIONAL SWAY ANALYSIS
                # ml_sway: Medio-lateral (side-to-side) sway variability
                # Standard deviation of horizontal movement
                # Excessive ML sway is strongly associated with fall risk
                features['ml_sway'] = np.std(hip_trajectory[:, 0])
                
                # ap_sway: Anterior-posterior (forward-backward) sway variability
                # Standard deviation of forward-backward movement
                # AP sway indicates balance control in the sagittal plane
                features['ap_sway'] = np.std(hip_trajectory[:, 1])
        
        return features
    
    def _calculate_joint_coordination(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate multi-joint coordination features.
        
        Joint coordination analyzes how well different joints work together
        during movement. Poor coordination between joints can indicate
        neurological issues, musculoskeletal problems, or increased fall risk.
        """
        features = {}
        
        # JOINT ANGLE ANALYSIS
        # Calculate angles for all major joints over time
        joint_angles = self._calculate_all_joint_angles(df)
        
        if joint_angles:
            # ADJACENT JOINT COORDINATION
            # Analyze coordination between joints that are mechanically linked
            coordination_pairs = [
                ('left_knee', 'left_ankle'),     # Lower limb kinetic chain
                ('right_knee', 'right_ankle'),   # Lower limb kinetic chain
                ('left_shoulder', 'left_elbow'), # Upper limb kinetic chain
                ('right_shoulder', 'right_elbow') # Upper limb kinetic chain
            ]
            
            for joint1, joint2 in coordination_pairs:
                if joint1 in joint_angles and joint2 in joint_angles:
                    angles1 = joint_angles[joint1]
                    angles2 = joint_angles[joint2]
                    
                    if len(angles1) > 0 and len(angles2) > 0:
                        min_len = min(len(angles1), len(angles2))
                        # joint_coordination: Cross-correlation between adjacent joint angles
                        # Values range from -1 to 1
                        # High positive values indicate synchronized movement
                        # Values near 0 indicate poor coordination
                        # Used to assess kinetic chain function
                        correlation = np.corrcoef(angles1[:min_len], angles2[:min_len])[0, 1]
                        features[f'{joint1}_{joint2}_coord'] = correlation if not np.isnan(correlation) else 0
        
        return features
    
    def _calculate_limb_correlation(self, df: pd.DataFrame, limb1: str, limb2: str) -> float:
        """Calculate cross-correlation between two limbs."""
        limb1_data = df[df['keypoint_idx'].isin(self.body_parts[limb1])]
        limb2_data = df[df['keypoint_idx'].isin(self.body_parts[limb2])]
        
        if limb1_data.empty or limb2_data.empty:
            return 0
        
        # Vectorised: mean position per timestamp, then frame-to-frame velocity magnitude
        l1_pos = limb1_data.groupby('timestamp', sort=True)[['x', 'y', 'z']].mean().dropna()
        l2_pos = limb2_data.groupby('timestamp', sort=True)[['x', 'y', 'z']].mean().dropna()
        
        # Align on common timestamps
        common = l1_pos.index.intersection(l2_pos.index)
        if len(common) < 3:
            return 0
        
        l1_vel = np.linalg.norm(np.diff(l1_pos.loc[common].values, axis=0), axis=1)
        l2_vel = np.linalg.norm(np.diff(l2_pos.loc[common].values, axis=0), axis=1)
        
        if len(l1_vel) < 2:
            return 0
        correlation = np.corrcoef(l1_vel, l2_vel)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def _calculate_phase_coupling(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate phase coupling between limbs.
        
        Phase coupling analyzes the temporal relationships between limb movements.
        It measures how synchronized the oscillatory patterns of different limbs are,
        which is important for coordinated movement and gait analysis.
        """
        features = {}
        
        # LIMB PHASE ANALYSIS
        # Analyze phase relationships between major limbs
        limbs = ['left_arm', 'right_arm', 'left_leg', 'right_leg']
        
        limb_phases = {}
        for limb in limbs:
            limb_data = df[df['keypoint_idx'].isin(self.body_parts[limb])]
            if not limb_data.empty:
                # Vectorised: mean x/y/z per timestamp → frame-to-frame velocity magnitude
                limb_mean = limb_data.groupby('timestamp', sort=True)[['x', 'y', 'z']].mean().dropna()
                if len(limb_mean) > 2:
                    velocities = np.linalg.norm(np.diff(limb_mean.values, axis=0), axis=1)
                    if len(velocities) > 1:
                        phases = np.arctan2(np.gradient(velocities), velocities)
                        limb_phases[limb] = phases
        
        # PHASE COUPLING STRENGTH CALCULATION
        # Calculate coupling between all limb pairs
        for i, limb1 in enumerate(limbs):
            for limb2 in limbs[i+1:]:
                if limb1 in limb_phases and limb2 in limb_phases:
                    phases1 = limb_phases[limb1]
                    phases2 = limb_phases[limb2]
                    
                    min_len = min(len(phases1), len(phases2))
                    if min_len > 5:
                        # phase_coupling: Strength of phase synchronization
                        # Calculated using complex exponentials to measure phase coherence
                        # Values range from 0 (no coupling) to 1 (perfect coupling)
                        # Higher values indicate better inter-limb coordination
                        phase_diff = phases1[:min_len] - phases2[:min_len]
                        coupling_strength = np.abs(np.mean(np.exp(1j * phase_diff)))
                        features[f'{limb1}_{limb2}_phase_coupling'] = coupling_strength
        
        return features
    
    def _calculate_center_of_pressure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate center of pressure related features.
        
        Center of pressure (COP) represents the weighted average location of
        pressure under the feet. COP movement patterns are crucial indicators
        of balance control and postural stability.
        """
        features = {}
        
        # FEET AND ANKLE KEYPOINTS FOR PRESSURE ESTIMATION
        # Use feet keypoints as pressure points to estimate COP
        feet_keypoints = [27, 28, 29, 30, 31, 32]  # ankles and feet
        feet_data = df[df['keypoint_idx'].isin(feet_keypoints)]
        
        if not feet_data.empty:
            # Vectorised confidence-weighted COP per timestamp
            fd = feet_data.dropna(subset=['x', 'y']).copy()
            fd['wx'] = fd['x'] * fd['confidence']
            fd['wy'] = fd['y'] * fd['confidence']
            g = fd.groupby('timestamp')
            sum_w = g['confidence'].sum()
            cop_x = (g['wx'].sum() / sum_w).sort_index()
            cop_y = (g['wy'].sum() / sum_w).sort_index()
            cop_trajectory = np.column_stack([cop_x.values, cop_y.values])
            
            if len(cop_trajectory) > 5:
                cop_array = cop_trajectory
                
                # COP MOVEMENT CHARACTERISTICS
                # cop_velocity: Average velocity of center of pressure movement
                # Higher values indicate more rapid balance adjustments
                # Calculated as mean of frame-to-frame displacement magnitudes
                features['cop_velocity'] = np.mean(np.linalg.norm(np.diff(cop_array, axis=0), axis=1))
                
                # cop_acceleration: Average acceleration of COP movement
                # High accelerations indicate sudden balance corrections
                # Calculated as mean of second-order differences
                features['cop_acceleration'] = np.mean(np.linalg.norm(np.diff(cop_array, n=2, axis=0), axis=1))
                
                # cop_path_length: Total path length traveled by COP
                # Longer paths indicate more postural adjustments
                # Sum of all frame-to-frame displacements
                features['cop_path_length'] = np.sum(np.linalg.norm(np.diff(cop_array, axis=0), axis=1))
        
        return features
    
    def _calculate_sway_area(self, trajectory: np.ndarray) -> float:
        """
        Calculate the area of postural sway.
        
        Sway area represents the area covered by the center of gravity trajectory.
        Larger areas indicate greater postural instability and increased fall risk.
        """
        if len(trajectory) < 3:
            return 0
        
        # Use convex hull area for precise sway area calculation
        # If convex hull fails, use standard deviations as approximation
        try:
            hull = ConvexHull(trajectory)
            return hull.volume  # In 2D, volume is area
        except Exception:
            # Fallback: approximate area as product of X and Y standard deviations
            return np.std(trajectory[:, 0]) * np.std(trajectory[:, 1])
    
    def _calculate_sway_velocity(self, trajectory: np.ndarray) -> float:
        """
        Calculate mean sway velocity.
        
        Sway velocity measures how fast the center of gravity moves.
        Higher velocities indicate more rapid postural adjustments,
        which may suggest poor balance control.
        """
        if len(trajectory) < 2:
            return 0
        
        # Calculate frame-to-frame displacements and their magnitudes
        velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        return np.mean(velocities)
    
    def _calculate_sway_acceleration(self, trajectory: np.ndarray) -> float:
        """
        Calculate mean sway acceleration.
        
        Sway acceleration measures the rate of change of sway velocity.
        High accelerations indicate sudden, jerky movements often associated
        with balance corrections and increased fall risk.
        """
        if len(trajectory) < 3:
            return 0
        
        # Calculate second-order differences (acceleration) and their magnitudes
        accelerations = np.linalg.norm(np.diff(trajectory, n=2, axis=0), axis=1)
        return np.mean(accelerations)
    
    def _calculate_all_joint_angles(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Calculate all joint angles over time.
        
        Joint angles are fundamental biomechanical measures that indicate
        joint function and movement quality. Abnormal angle patterns
        can indicate musculoskeletal issues or movement compensations.
        """
        joint_angles = {}
        
        # JOINT ANGLE DEFINITIONS
        # Each joint angle is defined by three keypoints: (point1, vertex, point2)
        angle_definitions = {
            'left_knee': (23, 25, 27),
            'right_knee': (24, 26, 28),
            'left_elbow': (11, 13, 15),
            'right_elbow': (12, 14, 16),
            'left_shoulder': (13, 11, 23),
            'right_shoulder': (14, 12, 24),
            'left_ankle': (25, 27, 29),
            'right_ankle': (26, 28, 30)
        }
        
        # Vectorised: for each joint, filter the relevant three keypoints once,
        # pivot to wide format, then compute angles row-by-row (no per-frame df filter)
        for joint_name, (p1, p2, p3) in angle_definitions.items():
            kp_subset = df[df['keypoint_idx'].isin([p1, p2, p3])].copy()
            if kp_subset.empty:
                continue
            
            # Pivot: rows=timestamp, cols=(coord, keypoint_idx)
            try:
                wide = kp_subset.pivot_table(
                    index='timestamp', columns='keypoint_idx',
                    values=['x', 'y', 'z'], aggfunc='first'
                )
            except Exception:
                continue
            
            angles = []
            for _, row in wide.iterrows():
                try:
                    pt = {}
                    for kp in [p1, p2, p3]:
                        x = row[('x', kp)]
                        y = row[('y', kp)]
                        z = row[('z', kp)]
                        if pd.isna(x) or pd.isna(y) or pd.isna(z):
                            break
                        pt[kp] = np.array([x, y, z])
                    else:
                        angles.append(self._calculate_angle(pt[p1], pt[p2], pt[p3]))
                except (KeyError, TypeError):
                    continue
            
            if angles:
                joint_angles[joint_name] = angles
        
        return joint_angles
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle between three points (p2 is the vertex).
        
        This method calculates the angle at point p2 formed by vectors p2->p1 and p2->p3.
        Used for measuring joint angles in biomechanical analysis.
        
        Args:
            p1: First point coordinates [x, y, z]
            p2: Vertex point coordinates [x, y, z] 
            p3: Third point coordinates [x, y, z]
            
        Returns:
            Angle in degrees (0-180°)
        """
        # Create vectors from vertex to the other two points
        v1 = p1 - p2  # Vector from p2 to p1
        v2 = p3 - p2  # Vector from p2 to p3
        
        # Handle zero vectors (points are coincident)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        
        # Calculate angle using dot product formula: cos(θ) = (v1·v2)/(|v1||v2|)
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        # Clamp to valid range to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Convert from radians to degrees
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def extract_advanced_features(self, df: pd.DataFrame, base_features: Dict[str, float]) -> Dict[str, float]:
        """
        Extract advanced features for a single subject.
        
        This is the main method that orchestrates the extraction of all advanced features.
        It combines temporal, frequency, biomechanical, and interaction features to create
        a comprehensive feature set for fall risk prediction.
        
        Args:
            df: Preprocessed keypoint DataFrame for one subject
            base_features: Basic features already extracted (velocity, angles, etc.)
            
        Returns:
            Dictionary of advanced features with feature names as keys
        """
        advanced_features = {}
        
        # TEMPORAL SEQUENCE FEATURES
        # Extract features related to movement patterns over time
        try:
            temporal_features = self._calculate_temporal_sequences(df)
            advanced_features.update(temporal_features)
        except Exception as e:
            print(f"Error in temporal features: {e}")
        
        # FREQUENCY DOMAIN FEATURES  
        # Extract features from frequency analysis of movement patterns
        try:
            frequency_features = self._calculate_frequency_domain_features(df)
            advanced_features.update(frequency_features)
        except Exception as e:
            print(f"Error in frequency features: {e}")
        
        # BIOMECHANICAL FEATURES
        # Extract features related to body mechanics and postural control
        try:
            biomech_features = self._calculate_biomechanical_features(df)
            advanced_features.update(biomech_features)
        except Exception as e:
            print(f"Error in biomechanical features: {e}")
        
        # INTERACTION FEATURES
        # Extract features that capture relationships between basic features
        try:
            interaction_features = self._calculate_interaction_features(base_features)
            advanced_features.update(interaction_features)
        except Exception as e:
            print(f"Error in interaction features: {e}")
        
        # DATA QUALITY ASSURANCE
        # Replace any NaN or infinite values with 0 to ensure model compatibility
        for key, value in advanced_features.items():
            if pd.isna(value) or np.isinf(value):
                advanced_features[key] = 0
        
        return advanced_features


def extract_enhanced_features_for_all_subjects(keypoint_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract enhanced features (basic + advanced) for all subjects.
    
    Args:
        keypoint_data: Dictionary mapping subject IDs to their keypoint DataFrames
        
    Returns:
        DataFrame with enhanced features for all subjects
    """
    # Import basic feature extraction from data_preprocessing module
    # The file is named '2. data_preprocessing.py' so normal import fails; use importlib
    import importlib.util as _ilu
    import os as _os
    try:
        _spec = _ilu.spec_from_file_location(
            "data_preprocessing",
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "2. data_preprocessing.py")
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        extract_features_for_subject = _mod.extract_features_for_subject
    except Exception as _e:
        print(f"Warning: Could not import basic feature extraction ({_e}), using mock basic features")
        extract_features_for_subject = None
    
    advanced_extractor = AdvancedFeatureExtractor()
    
    all_features = []
    
    for subject_id, df in tqdm(keypoint_data.items(), desc="Extracting enhanced features"):
        try:
            # Extract basic features first
            if extract_features_for_subject is not None:
                basic_features = extract_features_for_subject(df)
            else:
                # If basic feature extraction is not available, create minimal set
                basic_features = {
                    'mean_velocity': 0.0,
                    'std_velocity': 0.0,
                    'foot_distance_std': 0.0,
                    'knee_angles_mean': 0.0,
                    'core_mean_velocity': 0.0,
                    'z_range': 0.0,
                    'x_mean': 0.0,
                    'mean_confidence': 0.0
                }
            
            # Extract advanced features
            advanced_features = advanced_extractor.extract_advanced_features(df, basic_features)
            
            # Combine all features
            all_features_dict = {**basic_features, **advanced_features}
            all_features_dict['Subject_ID'] = subject_id
            all_features.append(all_features_dict)
            
        except Exception as e:
            print(f"Error extracting enhanced features for {subject_id}: {e}")
    
    features_df = pd.DataFrame(all_features)
    
    # Reorder columns to have Subject_ID first
    cols = ['Subject_ID'] + [col for col in features_df.columns if col != 'Subject_ID']
    features_df = features_df[cols]
    
    print(f"Extracted {len(features_df.columns)-1} enhanced features for {len(features_df)} subjects")
    
    return features_df


if __name__ == "__main__":
    # Test enhanced feature extraction
    import importlib.util as _ilu, os as _os
    _spec = _ilu.spec_from_file_location(
        "data_preprocessing",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "2. data_preprocessing.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    load_and_preprocess_data = _mod.load_and_preprocess_data
    
    keypoint_data, labels_df = load_and_preprocess_data()
    enhanced_features_df = extract_enhanced_features_for_all_subjects(keypoint_data)
    
    print("\nEnhanced feature extraction completed!")
    print(f"Features shape: {enhanced_features_df.shape}")
    print(f"Number of features: {len(enhanced_features_df.columns)-1}")
    
    # Show feature categories
    feature_names = [col for col in enhanced_features_df.columns if col != 'Subject_ID']
    print(f"\nTotal features extracted: {len(feature_names)}")
    
    # Sort by Subject_ID before saving
    enhanced_features_df = enhanced_features_df.sort_values('Subject_ID')
    
    # Save enhanced features
    output_path = "/Users/wk/Desktop/Testing/Dance2.0/processed_data/enhanced_features.csv"
    enhanced_features_df.to_csv(output_path, index=False)
    print(f"Enhanced features saved to: {output_path}")
