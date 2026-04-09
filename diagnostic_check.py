import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
import os

def run_diagnostics(npz_path="dataset_processed.npz"):
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found. Ensure the dataset is in the current directory.")
        return
        
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    X_audio = data['X_audio']
    Y_labels = data['Y_labels']
    Y_metadata = data['Y_metadata']
    
    if 'Y_groups' in data:
        Y_groups = data['Y_groups']
    else:
        Y_groups = None

    print(f"Loaded X_audio shape: {X_audio.shape}")
    print(f"Loaded Y_metadata shape: {Y_metadata.shape}")
    
    print("-" * 50)
    print("1. Visual Inspection (MFCC Heatmaps)")
    
    # We will search for a class that has both 'clean' and 'aug_heavy'
    # By default, assuming class '1' is 'stop' based on CLASS_MAP
    # Just grab any valid index for visualization
    stop_class_id = 1
    
    clean_idx = np.where((Y_metadata == 'clean') & (Y_labels == stop_class_id))[0]
    heavy_idx = np.where((Y_metadata == 'aug_heavy') & (Y_labels == stop_class_id))[0]
    
    if len(clean_idx) > 0 and len(heavy_idx) > 0:
        idx_clean = clean_idx[0]
        idx_heavy = heavy_idx[0]
        
        mfcc_clean = X_audio[idx_clean]
        mfcc_heavy = X_audio[idx_heavy]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(np.transpose(mfcc_clean), ax=axes[0], cmap='viridis')
        axes[0].set_title(f"Clean MFCC (Label: {stop_class_id}) - Idx: {idx_clean}")
        axes[0].set_xlabel("Time Frames")
        axes[0].set_ylabel("MFCC Coefficients")
        
        sns.heatmap(np.transpose(mfcc_heavy), ax=axes[1], cmap='viridis')
        axes[1].set_title(f"Aug Heavy MFCC (Label: {stop_class_id}) - Idx: {idx_heavy}")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("MFCC Coefficients")
        
        plt.tight_layout()
        plt.savefig("diagnostic_mfcc_comparison.png")
        print("-> Saved 'diagnostic_mfcc_comparison.png' for visual inspection.")
    else:
        print("Warning: Could not find both 'clean' and 'aug_heavy' samples for label 1.")

    print("-" * 50)
    print("2. Mathematical Variance Check")
    all_clean_idx = np.where(Y_metadata == 'clean')[0]
    all_heavy_idx = np.where(Y_metadata == 'aug_heavy')[0]
    
    if len(all_clean_idx) > 0 and len(all_heavy_idx) > 0:
        clean_mfccs = X_audio[all_clean_idx]
        heavy_mfccs = X_audio[all_heavy_idx]
        
        print(f"Clean samples ({len(all_clean_idx)}):")
        print(f"  Mean = {clean_mfccs.mean():.6f}")
        print(f"  Min  = {clean_mfccs.min():.6f}")
        print(f"  Max  = {clean_mfccs.max():.6f}")
        print(f"  Var  = {clean_mfccs.var():.6f}")
        
        print(f"Heavy 'aug' samples ({len(all_heavy_idx)}):")
        print(f"  Mean = {heavy_mfccs.mean():.6f}")
        print(f"  Min  = {heavy_mfccs.min():.6f}")
        print(f"  Max  = {heavy_mfccs.max():.6f}")
        print(f"  Var  = {heavy_mfccs.var():.6f}")
        
        if np.isclose(clean_mfccs.mean(), heavy_mfccs.mean(), atol=1e-5):
            print(">>> WARNING: The statistical distributions are nearly identical!")
            print(">>> The augmentation math likely failed or was skipped.")
    else:
        print("Missing either clean or heavy samples to compare mathematically.")

    print("-" * 50)
    print("3. Leakage & Split Verification")
    if Y_groups is not None:
        print("Simulating train_test_split using GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)...")
        # Ensure we mimic the initial split (Train/Val vs Test)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        
        # NOTE: If dataset_processed.npz contains ALL data, this mimics the split exactly
        train_idx, test_idx = next(gss.split(X_audio, groups=Y_groups))
        
        train_groups = set(Y_groups[train_idx])
        test_groups = set(Y_groups[test_idx])
        
        intersection = train_groups.intersection(test_groups)
        print(f"Unique base filenames in Train/Val: {len(train_groups)}")
        print(f"Unique base filenames in Test: {len(test_groups)}")
        print(f"Intersection between Train and Test sets: {len(intersection)}")
        
        if len(intersection) > 0:
            print(f">>> WARNING: DATA LEAKAGE DETECTED! Overlapping base filenames: {len(intersection)}")
        else:
            print("-> SUCCESS: No dataset leakage. Augmented variations do not bridge the train/test gap.")
    else:
        print("Y_groups array not found in npz file. Cannot verify split integrity.")
        
    print("-" * 50)
    print("4. Padding Bias Check")
    # Check 5 random samples for exact trailing zeros/identical padding values
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_audio), min(5, len(X_audio)), replace=False)
    
    print("Checking identical trailing frames for padding validation:")
    for i in sample_indices:
        sample_mfcc = X_audio[i]
        last_frame = sample_mfcc[-1]
        
        identical_frames = 0
        # Traverse backward
        for frame_idx in range(len(sample_mfcc)-1, -1, -1):
            if np.allclose(sample_mfcc[frame_idx], last_frame):
                identical_frames += 1
            else:
                break
        print(f"  Sample {i} (Label {Y_labels[i]}, Type: {Y_metadata[i]}): {identical_frames} identical trailing padding frames out of {len(sample_mfcc)} total frames.")

    # Calculate global temporal variance to see where true data lives
    time_variance = np.var(X_audio, axis=(0, 2))  # shape: (Time_Frames,)
    print("\nGlobal variance per time frame (across all samples/features):")
    # Present just the last 10 padding frames logic
    print("  Variance of the LAST 10 frames:", time_variance[-10:])

if __name__ == "__main__":
    run_diagnostics()
