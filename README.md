# License Plate Detection and Recognition

This github repository requires the creation of three (3) python/conda environments for the development of the project.
The name of the environment directories are:

```bash
LPDR-SYSTEM/label-env/
LPDR-SYSTEM/tf-fyp-env/
LPDR-SYSTEM/torch-fyp-env/
```

## Label Studio Software

The Label Studio software will be used for annotating the license plate on the dataset images. This software is used to create the training and testing dataset for the project.

This can be used for both the detection and segmentation aspect of the project.

Label Studio can be installed by:

```bash
pip install label-studio
```

Label Studio can be launched using:

```bash
label-studio start
```

## Tensorflow Environment

Using the tf-fyp-env python environment, we must install the dependencies necessary for this environment. Place the following in your requirements.txt file when installing modules in your tensorflow environment.

```requirements.txt
autopep8
pandas
numpy
matplotlib
opencv-python
scikit-learn
tensorflow
tensorflow-cpu
# Uncomment if GPU is installed in machine
# tensorflow-gpu

```

**Note**: Tensorflow requires the following to be installed:

- [GPU Drivers (version 450.80.02 or higher)](https://www.nvidia.com/download/index.aspx?lang=en-us)
- [CUDA 11.2](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN SDK 8.1.0](https://developer.nvidia.com/cudnn)

## PyTorch Environment

Using the torch-fyp-env, we must install the dependencies necessary for this environment.Place the following in your requirements.txt file when installing modules in your Pytorch environment.

**Note**: PyTorch requires [CUDA 11.7](https://developer.nvidia.com/cuda-toolkit-archive)

```requirements.txt
autopep8
pandas
numpy
matplotlib
opencv-python
scikit-learn
--extra-index-url https://download.pytorch.org/whl/cu117
torch
torchvision
torchaudio
```
