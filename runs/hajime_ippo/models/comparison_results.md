# Full Model Comparison: 6 Architectures

## Results

| Metric              | NatureCNN          | CNN_LSTM           | ViT_Transformer    | Impoola_CNN        | Impala_CNN         | ResNet18           |
|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| Final Loss          | 3.476316           | 3.024518           | 2.998286           | 2.933126           | 3.027630           | 2.901681           |
| Training Time       | 18.3            s | 67.8            s | 3288.8          s | 226.2           s | 345.1           s | 251.3           s |
| Model Size          | 16.33           MB | 23.71           MB | 9.70            MB | 4.20            MB | 67.95           MB | 44.37           MB |
| Num. Params         | 4,196,810         | 6,116,779         | 2,448,010         | 1,009,258         | 17,720,938        | 11,516,938        |

## Analysis

- **Lowest Loss**: ResNet18
- **Fastest**: NatureCNN
- **Lightest**: Impoola CNN

### Description of the Newly Added Architectures:
*   **Impala-CNN**: Classic model from the IMPALA paper. It has excellent spatial memorization capacity because it keeps the Flatten of the convolutional layer, but it produces a very heavy file.
*   **ResNet-18**: Deep convolutional architecture with standard residual connections. It offers good gradient stability, but it is redundant for 128x128 resolutions.

### Note on Swin Transformer and ConvNeXt:
Both the standard **Swin Transformer** and **ConvNeXt (Tiny)** were evaluated, but **removed from the active training script** due to extreme processing slowness and high hardware consumption. 
The ConvNeXt (Tiny) and the Swin Transformer each have about 28 million parameters, which causes a video memory bottleneck (VRAM Thrashing) when processing batches of 384 on stacked images, resulting in estimated training times of **40 hours** (in the ConvNeXt case) on the development hardware, which rules out agile local testing. Although they are extremely powerful architectures for general vision, they require cluster resources or GPUs with high dedicated memory.

## Experiment Configuration

- Training Epochs: 10
- Batch Size: 384
- Learning Rate: 0.0001
- Device: cuda

## Artifacts
- Models saved in the `./models/` folder
- MLflow Experiment: `Model_Comparison`
