# Regularized Neural Detection for Millimeter Wave Massive MIMO Communication Systems with One-bit ADCs

## Table of Contents
- [About the Project](#about-the-project)
- [mmW-ROBNet](#mmw-robnet)
- [Prerequisites](#prerequisites)
- [Running the Code](#running-the-code)
- [Acknowledgements](#acknowledgements)
- [Citation](#reference)

## About the Project

Our project addresses the challenge of multi-user massive MIMO signal detection from one-bit received measurements, with a specific emphasis on the intricacies of the wireless channel. While many existing approaches concentrate on detector design for rich-scattering, homogeneous Rayleigh fading channels, we introduce a novel approach for detecting signals in the context of lower diversity millimeter-wave (mmWave) channels.

### Key Objectives

The primary objectives of this project are as follows:

#### 1. Parametric Deep Learning System

We have developed the **mmW-ROBNet**, which is a parametric deep learning system designed to enhance signal detection in mmWave channels.

#### 2. Constellation-Aware Loss Function

We have devised a specialized loss function that is **constellation-aware**, enabling more accurate and effective signal detection.

#### 3. Hierarchical Detection Training Strategy
We implement a **hierarchical training strategy** to enhance the signal detection process.


### mmW-ROBNet

![mmW-ROBNet](https://github.com/adityasant/mmW_ROBNet/blob/main/proj_robnet_overall_mmwave.png)

The mmW-ROBNet is the key contribution of this project. Here's a brief summary of its features:

- **Unfolded DNN Algorithm:** The mmW-ROBNet is developed as an approach to solve constrained optimizations for general channel matrices via a DNN-augmented GD algorithm.

- **Channel-Specific Design:** The mmW-ROBNet framework incorporates the specific properties of the mmWave channel, making it suitable for the unique challenges of mmWave communications.

- **T-Stage Regularized GD:** The framework unfolds a T-stage regularized GD algorithm into T distinct sub-networks, each represented as a stage.

- **OBMNet Iteration:** At the beginning of each stage, the OBMNet iteration generates gradients and outputs, contributing to the signal detection process.

- **User-Matched Gradient:** The mmWave-channel powers per user are matched to the OBMNet-generated gradient, improving the accuracy of signal detection.

- **Regularization Network:** A regularization network fine-tunes the previous estimates, user-matched gradients, and OBMNet outputs.

- **Stage-Dependent Correction:** A residual link from the OBMNet output corrects the unconstrained OBMNet step in a stage-dependent manner.

- **Normalization:** The final output is normalized to ensure effective signal detection.


## Prerequisites

Before you get started, ensure you have met the following requirements:

- [PyTorch](https://pytorch.org/) version 1.12 or higher.
  - You can download PyTorch from the official [Docker container](https://hub.docker.com/r/pytorch/pytorch) (e.g., pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime).
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/) for plotting

You can install NumPy, SciPy, and Matplotlib using pip:

```shell
pip install numpy scipy matplotlib
```




## Running the Code

The following are the steps for running the code, specifically for training and testing the mmW-ROBNet.

1. **Data Generation:** The Python file `create_data_random_chan.py` contains various functions used for data generation and preprocessing. These functions are essential for creating the channel matrix, sorting users, and generating measurements.

2. **Training:** To train the one-bit mmW-ROBNet, use the training script `train_one_bit_mmW_ROBNet.py`. You can run the training process using the following command:

   ```shell
   python train_one_bit_mmW_ROBNet.py
   ```
   The trained network will be saved in the Saved_Networks folder. During training, the code will save checkpoint files for the trained network at specific intervals, enabling you to resume training from the last saved checkpoint.

3. **Testing:** For testing the network's performance, use the testing script test_script_mmW_ROBNet.py. Run the testing script using the following command:

   ```shell
   python test_script_mmW_ROBNet.py
   ```
   The Bit Error Rate (BER) results will be saved in the BER_Results folder. During testing, the code will save BER results for different signal-to-noise ratios (SNR) and users.


## Acknowledgements
The authors would like to acknowledge support from PRP - Nautilus for access to high speed GPU resources for training and testing of our scripts. This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019, the University of California Office of the President, and the University of California San Diego's California Institute for Telecommunications and Information Technology/Qualcomm Institute. Thanks to CENIC for the 100Gbps networks.


## Reference

If you use this code or find this work helpful in your research, please cite the following paper:

Sant, A. and Rao, B.D., 2023, June. Regularized Neural Detection for Millimeter Wave Massive MIMO Communication Systems with One-Bit ADCs. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE.


