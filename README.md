### Dynamic Frequency Domain Curriculum Learning (DFCL) for Deepfake Detection

This repository provides the official implementation of the Dynamic Frequency Domain Curriculum Learning (DFCL) framework, as described in the paper:
Dynamic Frequency Domain Curriculum Learning: A Novel Framework for Adaptive Image Forgery Detection

### Introduction
DFCL is a novel framework for deepfake detection that leverages frequency domain analysis and curriculum learning to enhance model generalization. By dynamically scheduling training samples from "easy-to-hard" based on frequency domain difficulty scores, DFCL improves cross-domain detection performance on various deepfake datasets.

### Dataset Preparation
Use the DeepfakeBench dataset preparation script to download and preprocess the following datasets:
- FF++ (FaceForensics++)
- Celeb-DFv1/v2
- DFDC (DeepFake Detection Challenge)
- DFDCP (DFDC Preview)
- UADFV

Our work is based on deepfakebench, just place the file in the corresponding location.
