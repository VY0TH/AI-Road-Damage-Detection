# Road Damage Segmentation Using CNN and Spatial LSTM

## 1. Project Overview

This project investigates road-damage segmentation from video frames
using a hybrid Convolutional Neural Network (CNN) and Spatial Long
Short-Term Memory (LSTM) architecture.

The CNN extracts spatial features from each frame. The LSTM processes
features over time at individual spatial locations. A decoder combines
the spatial and temporal features to produce a binary damage mask.

The model predicts damage regions; it does not classify damage into
separate categories such as potholes and longitudinal cracks.

## 2. Dataset

The project uses the Pothole Videos dataset, containing paired RGB
videos and mask videos.

The processed subset contains:

| Split | Videos |
|---|---:|
| Training | 40 |
| Validation | 10 |
| Test | 10 |
| Total | 60 |

The original train, validation and test directories are preserved.
Each selected video contributes eight sampled frames.

## 3. Data Preprocessing

1. Read matching RGB and mask video files.
2. Check that frame count, FPS and resolution match.
3. Sample eight frame indices across each video.
4. Resize RGB frames to 128 x 128 pixels.
5. Convert RGB frames to RGB channel order.
6. Convert mask frames to grayscale.
7. Apply a threshold of 128 to create binary masks.
8. Resize masks with nearest-neighbour interpolation.
9. Store processed frames and masks in compressed NPZ files.

RGB frames are normalized to the range [0, 1] during training and
evaluation. Masks contain binary values 0 and 1.

Matching metadata does not independently prove that the RGB and mask
content is perfectly synchronized. This remains a dataset assumption
that should be checked visually.

## 4. Model Architecture

### 4.1 CNN Feature Extractor

The CNN applies convolution, batch normalization and ReLU activation
to extract spatial features from each frame. The feature maps are
reduced spatially by the initial strided convolution layers.

### 4.2 Spatial LSTM

For every spatial location in the CNN feature maps, the model constructs
a temporal feature sequence across the eight sampled frames.

An LSTM processes each sequence to capture temporal information.

### 4.3 Feature Fusion and Decoder

The spatial CNN features and temporal LSTM features are concatenated.
A convolutional decoder produces a one-channel segmentation logit map.
Bilinear interpolation restores the output to the input frame size.

A sigmoid function converts logits to probabilities. A threshold of
0.5 is used for the reported test evaluation.

## 5. Training Configuration

| Setting | Value |
|---|---|
| Device | CPU |
| Batch size | 1 |
| Maximum epochs | 20 |
| Learning rate | 0.0002 |
| Positive class weight | 3.0 |
| Optimizer | Adam |
| Loss | Weighted binary cross-entropy plus Dice loss |
| Early stopping patience | 6 epochs |
| Model selection | Validation global pixel Dice |

The best V2 checkpoint was saved at epoch 17.

Checkpoint:
`best_cnn_lstm_spatial_v2.pt`

## 6. Evaluation Metrics

### Dice coefficient

Dice measures overlap between the predicted mask and the ground-truth
mask. Higher values indicate better overlap.

### Intersection over Union

IoU measures the intersection divided by the union of predicted and
ground-truth foreground pixels.

### Pixel accuracy

Pixel accuracy measures the proportion of all pixels classified
correctly. It should not be considered alone because background pixels
can influence this metric substantially.

Both global pixel metrics and mean per-video metrics are reported.

## 7. Final Test Results

The final evaluation used ten held-out test videos with a threshold
of 0.5.

| Metric | Test result |
|---|---:|
| Mean per-video Dice | 0.6958 |
| Mean per-video IoU | 0.5588 |
| Global pixel Dice | 0.7514 |
| Global pixel IoU | 0.6018 |
| Pixel accuracy | 0.8986 |
| Ground-truth foreground | 20.02% |
| Predicted foreground | 20.76% |

These results describe the selected test subset and should not be
interpreted as a guarantee of real-world performance.

## 8. Limitations

1. The experiment uses a small subset of 60 videos.
2. Performance varies between individual test videos.
3. Some predictions contain false positives and missed damage regions.
4. The model performs binary segmentation, not damage-type classification.
5. Metadata matching alone does not prove exact RGB-mask alignment.
6. Evaluation on additional roads, weather conditions and camera views
   is needed to assess generalization.
7. The results depend on the selected dataset and preprocessing choices.

## 9. Future Work

- Increase the number and diversity of training videos.
- Verify frame-by-frame alignment between RGB videos and masks.
- Inspect false-positive and false-negative regions systematically.
- Compare against a segmentation baseline such as U-Net.
- Investigate class imbalance and suitable loss functions.
- Evaluate on a larger independent test set.
- Add damage-type classification only after a suitable labeled dataset
  and classification model are implemented.

## 10. Conclusion

The project implements a CNN and Spatial LSTM pipeline for binary road
damage segmentation in video sequences. On the selected ten-video test
subset, the model achieved a global Dice score of 0.7514 and global
IoU of 0.6018. Further testing and error analysis are necessary before
drawing conclusions about performance on unseen roads.
