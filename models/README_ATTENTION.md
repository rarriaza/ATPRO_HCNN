ATTENTION ARCHITECTURE:

The architecture consists on 3 parts:

#### CC (coarse classifier). Components:
- ResNet Block: First two ConvBlocks of ResNet-50 (Input layer - "conv2_block3_out")
    - Input: Image (32x32x3)
    - Output1 (Feature Map): RO1 feature matrix (8x8x256)
    - Output2 (Prediction): 20 softmax units (Coarse labels)
    
#### Attention Block: 
- Creates heatmap of the images and crops the input image with it:
    - Input: CC's Output1 (8x8x256) + Input image (original)
    - Output: Cropped Image (32x32x3)

#### FC (fine classifier) .Consists of:
- ResNet Block: First two ConvBlocks of ResNet-50 (Input layer - "conv2_block3_out")
    - Input1: Cropped Image (32x32x3) <- from attention block
    - Input2: CC's Output2 (Coarse prediction)
    - Output: 100 softmax units (Fine labels)
