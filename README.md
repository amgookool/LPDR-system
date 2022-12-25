# License Plate Detection and Recognition

## Programming Environment Configurations

### Prerequisites

- [Anaconda](https://www.anaconda.com/)

- [CMake](https://cmake.org/download/)

### Building OpenCV with CUDA support

#### **Windows**

- [Video Reference](https://www.youtube.com/watch?v=d8Jx6zO1yw0&t=306s)

##### **Requirements**

- [OpenCV Github Repository](https://github.com/opencv/opencv)

- [OpenCV Contrib Github Repository](https://github.com/opencv/opencv_contrib/tree/4.x)

- [Visual Studio 2019 Community](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=community&rel=16&utm_medium=microsoft&utm_campaign=download+from+relnotes&utm_content=vs2019ga+button)

##### **Notes**

- [CUDA Wikipedia (Version of GPU)](https://en.wikipedia.org/wiki/CUDA)

When using CMake to build the source code, the following flags must be set:

- **WITH_CUDA**
- **ENABLE_FAST_MATH**
- **OPENCV_DNN_CUDA**
- **OPENCV_EXTRA_MODULES_PATH** -> Set to path of the modules folder in the opencv_contrib folder
- **CUDA_FAST_MATH**
- **CUDA_ARCH_BIN** -> Refer to Wikipedia Link for CUDA version of your specific GPU (RTX 2060 is 7.5)
- **CMAKE_CONFIGURATION_TYPES** -> Set to Release

Ensure that python environment variables are set correctly for your anaconda distribution.
![OpenCV-Python Anaconda Path Settings](https://i.postimg.cc/zXgQgQBz/Open-CV-Python-Cmake-Configs.png)

Once configuration hve been set via the CMake GUI, go into your terminal and run the following command in your anaconda base environment:

```bash
cmake --build "C:\OpenCV-4.5.2\build" --target INSTALL --config Release
```

#### **Linux**

- [Article Reference 1](https://towardsdev.com/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367)

- [Article Reference 2](https://www.sproutworkshop.com/2022/06/how-to-compile-opencv-4-6-0-dev-with-cuda-11-7-and-cudnn-8-4-1-on-ubuntu-22-04/)

### **Tensorflow Environment**

Create a conda environment with a python version of 3.10.6

```conda
conda create --name <env_name> python=3.10.6
```

Run the following commands in you conda environment.

```bash
conda install -c conda-forge gcc=12.1.0
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install --upgrade pip

pip install tensorflow
```

Verify the Install using command:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

python3 -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
```

### **Pytorch Environment**

Create a conda environment with a python version of 3.10.6

```conda
conda create --name <env_name> python=3.10.6
```

Run the following commands in you conda environment.

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

#### **OR**

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Verify the Install using command:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"

python3 -c "import torch; print(torch.cuda.device_count())"

python3 -c "import torch; print(torch.cuda.current_device())"

python3 -c "import torch; print(torch.cuda.device(0))"

python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Annotating Data (Label Studio Software)

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
