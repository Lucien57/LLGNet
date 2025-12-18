# util/aug.py
import numpy as np
import torch
from typing import Tuple


class SegmentationReconstruction:
    """
    Segmentation and Reconstruction (S&R) data augmentation for EEG signals
    As described in the EEG Conformer paper

    Method: Divide training samples of the same category into Ns segments,
            then randomly concatenate while maintaining the original time order.
    """

    def __init__(self, n_segments: int = 8):
        self.n_segments = n_segments

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply S&R augmentation to a batch of EEG data

        Args:
            x: EEG data tensor of shape [batch_size, 1, channels, timepoints]
            y: Labels tensor of shape [batch_size]

        Returns:
            Augmented x and y
        """
        batch_size, _, n_channels, n_times = x.shape

        # For each class, create augmented samples
        unique_classes = torch.unique(y)
        augmented_x = []
        augmented_y = []

        for class_label in unique_classes:
            # Get indices of samples belonging to this class
            class_mask = (y == class_label)
            class_x = x[class_mask]
            class_y = y[class_mask]

            n_class_samples = class_x.shape[0]

            if n_class_samples < 2:
                # Need at least 2 samples for augmentation
                continue

            # Create augmented samples for this class
            for _ in range(n_class_samples):  # Create same number of augmented samples
                # Randomly select samples to segment
                sample_indices = torch.randperm(n_class_samples)[:self.n_segments]

                # Divide each selected sample into segments
                segments = []
                for idx in sample_indices:
                    sample = class_x[idx]  # [1, channels, timepoints]

                    # Calculate segment length
                    segment_length = n_times // self.n_segments

                    # Create segments
                    for seg_idx in range(self.n_segments):
                        start_idx = seg_idx * segment_length
                        end_idx = start_idx + segment_length

                        if seg_idx == self.n_segments - 1:  # Last segment takes remaining
                            end_idx = n_times

                        segment = sample[:, :, start_idx:end_idx]
                        segments.append(segment)

                # Randomly shuffle segments
                seg_indices = torch.randperm(len(segments))
                shuffled_segments = [segments[i] for i in seg_indices]

                # Concatenate segments to form new sample
                # Pad if necessary to match original length
                current_length = 0
                concatenated_segments = []

                for segment in shuffled_segments:
                    seg_len = segment.shape[-1]
                    if current_length + seg_len <= n_times:
                        concatenated_segments.append(segment)
                        current_length += seg_len
                    else:
                        # If we exceed the original length, take only part of the segment
                        remaining = n_times - current_length
                        if remaining > 0:
                            concatenated_segments.append(segment[:, :, :remaining])
                        break

                # Concatenate all segments
                if concatenated_segments:
                    new_sample = torch.cat(concatenated_segments, dim=-1)

                    # Pad if necessary
                    if new_sample.shape[-1] < n_times:
                        padding = n_times - new_sample.shape[-1]
                        new_sample = torch.nn.functional.pad(new_sample, (0, padding))

                    augmented_x.append(new_sample)
                    augmented_y.append(class_label)

        if augmented_x:
            augmented_x = torch.stack(augmented_x)
            augmented_y = torch.tensor(augmented_y, device=y.device, dtype=y.dtype)

            # Combine with original data
            combined_x = torch.cat([x, augmented_x], dim=0)
            combined_y = torch.cat([y, augmented_y], dim=0)

            return combined_x, combined_y

        return x, y


class EEGDataAugmentation:
    """Wrapper class for EEG data augmentation methods"""

    def __init__(self, augmentation_config: dict):
        self.augmentation_config = augmentation_config
        self.augmentations = []

        if augmentation_config.get('use_sr_augmentation', False):
            n_segments = augmentation_config.get('sr_n_segments', 8)
            self.augmentations.append(SegmentationReconstruction(n_segments))

    def apply(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all enabled augmentations"""
        if not self.augmentations:
            return x, y

        augmented_x, augmented_y = x, y
        for aug in self.augmentations:
            augmented_x, augmented_y = aug(augmented_x, augmented_y)

        return augmented_x, augmented_y

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply(x, y)


def create_augmentation(augmentation_config: dict):
    """Factory function to create augmentation instance"""
    return EEGDataAugmentation(augmentation_config)