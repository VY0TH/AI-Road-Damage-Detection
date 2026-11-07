r"""
====================================================================================================
SMOTE: SYNTHETIC MINORITY OVER-SAMPLING TECHNIQUE FOR IMBALANCED ROAD DAMAGE
====================================================================================================
File: techniques/smote/smote_balancer.py
Path: c:/Users/Vyom Thakur/Desktop/AI hackathon/RoadDamageProject/techniques/smote/smote_balancer.py

1. WHY SMOTE IS CRITICAL IN ROAD DAMAGE DETECTION:
   In real-world road inspection datasets (such as RDD2022), class frequencies are notoriously
   imbalanced due to real-world physics and occurrence rates:
     - Majority Class D00 (Longitudinal Cracks): 691 training samples (Dominant)
     - D10 (Transverse Cracks):                  340 training samples
     - D20 (Alligator Cracking):                 176 training samples
     - D40 (Potholes):                           121 training samples
     - Rare Class D44:                            50 training samples
     - Rare Class D50:                            32 training samples
     - Extreme Minority D43:                      14 training samples
     - Extreme Minority D01:                       8 training samples
     - Singular Sample D11:                        1 training sample
     - Singular Sample D0w0:                       1 training sample
   
   Standard cross-entropy loss gradients are overwhelmed by the dominant D00 class, causing
   the classifier to underfit or completely ignore rare classes (e.g., precision/recall of 0.00 for D01/D11).

2. THE MATHEMATICAL MECHANISM OF SMOTE:
   For every minority sample $\mathbf{x}_i \in S_{min}$:
     1. Compute Euclidean distance in feature space: $d(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_2$.
     2. Identify the $k$-nearest neighbors within the same minority class ($k \in [1, 5]$).
     3. Select a random neighbor $\mathbf{x}_{zi}$.
     4. Synthesize a new feature vector along the line segment joining $\mathbf{x}_i$ and $\mathbf{x}_{zi}$:
            $\mathbf{x}_{new} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}(0, 1)$

3. WHY DEEP FEATURE-SPACE SMOTE INSTEAD OF PIXEL-LEVEL SMOTE:
   Applying naive linear interpolation on raw pixels ($\mathbf{p}_{new} = \mathbf{p}_1 + \lambda(\mathbf{p}_2 - \mathbf{p}_1)$)
   causes ghosting, semi-transparent overlays, and destroys high-frequency asphalt textures.
   Instead, we apply SMOTE in the **latent feature space** extracted by our deep backbones
   (e.g., 1,280-dim EfficientNet embeddings or 2,048-dim ResNet-50 embeddings).
   In this latent manifold, linear combinations correspond to semantically valid distress combinations.
====================================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class LatentFeatureSMOTE:
    """
    Applies SMOTE oversampling in deep latent feature spaces to rectify severe road damage class imbalance.
    """

    def __init__(
        self,
        k_neighbors: int = 2,
        random_state: int = 42,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.class_names = class_names or [
            "D00", "D01", "D0w0", "D10", "D11",
            "D20", "D40", "D43", "D44", "D50"
        ]
        self.smote = None

    def balance_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        target_counts: Optional[Dict[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Synthesizes latent feature vectors for minority distress classes.
        
        Args:
            features (np.ndarray): Array of shape (N_samples, Feature_Dim)
            labels (np.ndarray): Array of integer class IDs (N_samples,)
            target_counts (Optional[Dict]): Custom target sampling strategy.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: Balanced features, balanced labels, audit stats.
        """
        unique, counts = np.unique(labels, return_counts=True)
        orig_dist = dict(zip(unique, counts))

        # Filter classes with >= 2 samples for k-NN interpolation; if class has 1 sample, duplicate once
        adjusted_features = list(features)
        adjusted_labels = list(labels)

        for cls_id, count in orig_dist.items():
            if count < (self.k_neighbors + 1):
                # Duplicate with slight jitter to enable k-NN graph construction
                needed = (self.k_neighbors + 1) - count
                idx = np.where(labels == cls_id)[0]
                for _ in range(needed):
                    src_idx = np.random.choice(idx)
                    jittered = features[src_idx] + np.random.normal(0, 0.01, features.shape[1])
                    adjusted_features.append(jittered)
                    adjusted_labels.append(cls_id)

        X_adj = np.array(adjusted_features)
        y_adj = np.array(adjusted_labels)

        # Apply imbalanced-learn SMOTE with safe k-neighbors
        effective_k = min(self.k_neighbors, min(np.unique(y_adj, return_counts=True)[1]) - 1)
        effective_k = max(1, effective_k)

        smote_engine = SMOTE(k_neighbors=effective_k, random_state=self.random_state)
        X_resampled, y_resampled = smote_engine.fit_resample(X_adj, y_adj)

        unique_after, counts_after = np.unique(y_resampled, return_counts=True)
        resampled_dist = dict(zip(unique_after, counts_after))

        stats = {
            "original_distribution": {self.class_names[c]: int(n) for c, n in orig_dist.items() if c < len(self.class_names)},
            "balanced_distribution": {self.class_names[c]: int(n) for c, n in resampled_dist.items() if c < len(self.class_names)},
            "original_total": len(labels),
            "resampled_total": len(y_resampled),
            "synthetic_samples_generated": len(y_resampled) - len(labels),
            "k_neighbors_used": effective_k,
        }

        return X_resampled, y_resampled, stats

    def plot_distribution_comparison(
        self,
        stats: Dict[str, Any],
        output_save_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> None:
        """
        Generates before-and-after bar chart illustrating the effect of SMOTE.
        """
        orig_dict = stats["original_distribution"]
        bal_dict = stats["balanced_distribution"]

        classes = sorted(list(set(orig_dict.keys()).union(set(bal_dict.keys()))))
        orig_counts = [orig_dict.get(c, 0) for c in classes]
        bal_counts = [bal_dict.get(c, 0) for c in classes]

        x = np.arange(len(classes))
        width = 0.38

        fig, ax = plt.subplots(figsize=(11, 5))
        rects1 = ax.bar(x - width/2, orig_counts, width, label='Before SMOTE (Severely Imbalanced)', color='#d62728', alpha=0.85)
        rects2 = ax.bar(x + width/2, bal_counts, width, label='After Latent SMOTE (Balanced)', color='#2ca02c', alpha=0.85)

        ax.set_ylabel('Number of Feature Samples', fontsize=11, fontweight='bold')
        ax.set_title(
            f"Class Distribution: Before vs. After SMOTE Resampling\n"
            f"Total Samples: {stats['original_total']} -> {stats['resampled_total']} "
            f"(+{stats['synthetic_samples_generated']} Synthetic Latent Embeddings)",
            fontsize=12, fontweight='bold', pad=12
        )
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')

        plt.tight_layout()

        if output_save_path:
            save_p = Path(output_save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_p), dpi=150, bbox_inches="tight")
            print(f"[SMOTE] Distribution comparison plot saved to: {save_p}")

        if show:
            plt.show()
        else:
            plt.close(fig)


# ==================================================================================================
# STANDALONE EXECUTION & VERIFICATION TEST
# ==================================================================================================
if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print("SMOTE BALANCER: VERIFICATION & SYNTHETIC RESAMPLING TEST")
    print(f"{'=' * 80}")

    # Simulate real-world RDD2022 crop distribution (Imbalanced: 10 classes)
    np.random.seed(42)
    feature_dim = 1280  # Matches EfficientNet-B0 latent bottleneck

    real_class_frequencies = [691, 8, 1, 340, 1, 176, 121, 14, 50, 32]
    simulated_features = []
    simulated_labels = []

    for cid, count in enumerate(real_class_frequencies):
        cls_center = np.random.randn(feature_dim)
        cls_feats = cls_center + np.random.normal(0, 0.2, (count, feature_dim))
        simulated_features.append(cls_feats)
        simulated_labels.extend([cid] * count)

    X_raw = np.vstack(simulated_features)
    y_raw = np.array(simulated_labels)

    print(f"Original Feature Matrix: {X_raw.shape} | Labels: {y_raw.shape}")

    balancer = LatentFeatureSMOTE(k_neighbors=2)
    X_bal, y_bal, audit = balancer.balance_features(X_raw, y_raw)

    print(f"Balanced Feature Matrix: {X_bal.shape} | Labels: {y_bal.shape}")
    print(f"Synthetic Samples Generated: {audit['synthetic_samples_generated']}")

    out_plot = PROJECT_ROOT / "results" / "diagnostics" / "smote_before_after_distribution.png"
    balancer.plot_distribution_comparison(audit, output_save_path=out_plot)
    print(f"{'=' * 80}\n")
