# ECE 661 Project 2: Input-Dependent Dynamic CNN Model

This is the course project for 2021F ECE 661 (Computer Engineering and Deep Neural Nets) at Duke University.
Read `proposal/proposal.pdf` for more detail.

Lead TA: Huanrui Yang

Students:
- Wei-Kai Liu
- Chung-Hsuan Tung
- Yi-Chen Chang
## Requirements
- Python (Tested with v3.7.10)
- PyTorch (Tested with v1.9.1)
## Dynamic Convolution

Apply dynamic convolution so the model can reach higher accuracy without much computational overhead.
Dynamic convolution is achieved by applying attention to multiple kernels in a channel. 

The command for dynamic convolution:
```
python3 dynamic.py <number of kernel>
```
The command for plotting attention map:
```
python3 plot_attention.py <model_path> <number of kernel>
```
  please make sure the number of kernel you key in is compatible with the model you load


## Channel Gating

Apply channel gating so a large model can reach higher accuracy with low FLOPs as a smaller model.
Also, apply batch-shaping for a similar function with batch normalization. 

## Adversarial Attack

Study and implement PGD adversarial attack on the above 2 models.
