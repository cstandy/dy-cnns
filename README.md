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

### Package Dependency

This project provides `Pipfile` to record the package dependency and it could
be used with `pipenv`.

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

Please make sure the number of kernel you key in is compatible with the model
you load.
The command for applying PGD attack is:

```
python3 pgd.py <select> 
```

| `select` | Meaning                  |
| :------: | ------------------------ |
| `1`      | Whitebox Attack          |
| `2`      | Adversarial Training     |
| `3`      | Transfer Blackbox Attack |


## Channel Gating

Apply channel gating so a large model can reach higher accuracy with low FLOPs as a smaller model.
Also, apply batch-shaping for a similar function with batch normalization.

The project provides 2 files under the folder `channel_gating/`.
`resnet-cifar10_cg_bs.ipynb` is a jupyter notebook file that includes all
the code and the description, and thus is recommended for understanding this
project.
`resnet-cifar10_cg_bs.py` is the python version of `resnet-cifar10_cg_bs.ipynb`,
and it allows you to train the model in the command line. Since the training
takes lots of time, you may want to train it in background with the python
file.

### Training

Run through `resnet-cifar10_cg_bs.ipynb` from the beginning from Setp 1 to Step
4.

### Test and Output Analysis

Run `resnet-cifar10_cg_bs.ipynb` but skip Step 4, continue on Step 5 to the end.
It will load model from `channel_gating/saved_model` and produce the output
analysis. The output analysis includes:

- Overall gate status across the validation set
- Frequency of gate opening for each classes in the CIFAR-10 dataset
- 2 example images of difference of gate opening frequency between classes
  - Deer vs. horse: Represent difference between similar classes
  - Airplane vs. cat: Represent difference between dissimilar classes

### Output Example

The analysis of channel gating could be found under `channel_gating/img`.

## Adversarial Attack

Study and implement PGD adversarial attack on the dynamic covolution models.
Please refer to [Dynamic Convolution](#dynamic-convolution).
