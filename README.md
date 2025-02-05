# Semi-Supervised Weed Detection Challenge

## Introduction
This repository contains the implementation of various semi-supervised learning approaches for weed detection in sesame crop fields. The project was developed as part of the *Semi-Supervised Weed Detection Challenge* organized by *4i Labs, KRITI'25, IITG*.

We aim to improve weed detection accuracy while minimizing the need for large labeled datasets. The challenge requires leveraging a small labeled dataset alongside a larger pool of unlabeled images to train an object detection model effectively.

## Problem Statement
The objective is to develop a weed detection model capable of identifying and localizing weeds in agricultural field images. The challenge dataset consists of:
- *Labeled Dataset:* 200 images with YOLOv8-format annotations
- *Unlabeled Dataset:* 1000 images without annotations
- *Test Dataset:* 100 images with annotations

Evaluation is based on the following metric:

Metric = 0.5 * (F1-Score) + 0.5 * (mAP@[.5:.95])


## Dataset
The dataset used in this project is provided by the challenge organizers. External datasets are strictly prohibited.

## Approaches Implemented
We implemented five different approaches for semi-supervised learning:

1. *Pseudo-Labeling Approach 1*
   - Gradual inclusion of pseudo-labeled data in training.
   - Confidence threshold starts at *0.85, reduced stepwise to **0.65*.
   - *Final Scores:* F1: 0.885, mAP@[.5:.95]: 0.602, Metric: 0.743

2. *Two-Stage Augmented Pseudo-Labeling*
   - Weak and strong augmentation applied to the unlabeled dataset.
   - Selectively includes high-confidence pseudo-labeled images.
   - *Final Scores:* F1: 0.912, mAP@[.5:.95]: 0.578, Metric: 0.745

3. *Pseudo-Labeling Approach 2*
   - Applies pseudo-labeling in a single step to all unlabeled data.
   - Best performance at confidence threshold *0.80*.
   - *Final Scores:* F1: 0.879, mAP@[.5:.95]: 0.608, Metric: 0.7438

4. *Augmentation of Labeled Data*
   - Applied image augmentations (flips, rotations, brightness variations).
   - Results were suboptimal due to overfitting and loss of quality.

5. *Mean Teacher Approach*
   - Utilizes student-teacher framework with exponential moving averages.
   - Computationally expensive with lower performance.
   - *Final Scores:* F1: 0.766, mAP@[.5:.95]: 0.457, Metric: 0.616

## Installation & Setup
### Prerequisites
- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Matplotlib

### Installation
Clone the repository:
sh
$ git clone https://github.com/YashK2803/WeedDetection_57.git
$ cd WeedDetection_57

Install dependencies:
sh
$ pip install -r requirements.txt


## Usage Instructions

   ```

## Results & Insights
- *Best approach:* *Pseudo-Labeling Approach 1* due to its better utilization of unlabeled data.
- *Challenges:* Data augmentation was not effective due to similarities in labeled and test images.
- *Future Work:* Implement advanced self-training techniques and refine augmentation strategies.