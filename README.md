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

## Building OpenCV with CUDA Support

[Video Reference](https://www.youtube.com/watch?v=d8Jx6zO1yw0&t=306s)

### Requirements

- [CMake](https://cmake.org/download/)

- [Anaconda](https://www.anaconda.com/)

- [OpenCV Github Repository](https://github.com/opencv/opencv)

- [OpenCV Contrib Github Repository](https://github.com/opencv/opencv_contrib/tree/4.x)

- [Visual Studio Community](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=community&rel=16&utm_medium=microsoft&utm_campaign=download+from+relnotes&utm_content=vs2019ga+button)

### Notes

- [CUDA Wikipedia (Version of GPU)](https://en.wikipedia.org/wiki/CUDA)

When using CMake to build the source code, the following flags must be set:

- **WITH_CUDA**
- **ENABLE_FAST_MATH**
- **BUILD_opencv_world**
- **OPENCV_EXTRA_MODULES_PATH** -> Set to path of the modules folder in the opencv_contrib folder
- **CUDA_FAST_MATH**
- **CUDA_ARCH_BIN** -> Refer to Wikipedia Link for CUDA version of your specific GPU (RTX 2060 is 7.5)
- **CMAKE_CONFIGURATION_TYPES** -> Set to Release

Ensure that python environment variables are set correctly for your anaconda distribution.
![OpenCV-Python Anaconda Path Settings](https://i.postimg.cc/zXgQgQBz/Open-CV-Python-Cmake-Configs.png)

Once configuration hve been set via the CMake GUI, go into your terminal and run the following command in your anaconda base environment:

```bash
cmake --build "C:/Users/amgoo/OpenCV-GPU/build" --target INSTALL --config Release
```
