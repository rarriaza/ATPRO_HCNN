## Algorithm

### 1. Define the Coarse Classifier (CC) and the Fine Classifier (FC)

#### CC consists of:
- ResNet Block: Layers 0 - 30 of ResNet-50
    - Input: Image (32x32x3)
    - Output: RO1 feature matrix
- Attention Block: Create heatmap of the images 
    - Input: Output RO1 (8x8x256)
    - Output: Input  RI1 (8x8x256)
- Prediction Layer:
    - Input: Output RO1 (8x8x256)
    - Output: 20 softmax units (Coarse labels)

#### FC consists of:
- ResNet Block: Layers 30 - 50 of ResNet-50
    - Input: RI1 + 20 coarse prediction labels
    - Output: RO2 feature matrix
- Prediction Bloack:
    - Input: RO2
    - Output: 100 softmax units (Fine labels)

### 2. Train CC

### 3. Freeze CC, Train FC

### 4. Fine-tune CC + FC together


#### ATPRO_HCNN
Implementation of two new versions of Hierarchical convolutional neural networks. 
These two versions have as upgrades: Implementation of hard attention and Uncertainty Estimation.

This implementatios were built on top of the architechture HD-CNN implemented by Zhicheng Yan et. al 2015 (https://arxiv.org/abs/1410.0736)
