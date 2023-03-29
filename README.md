# License Plate Detection and Recognition (LPDR) System

This repository contains the source code for the developed LPDR system for the final year project License Plate Detection and Recognition Using GPUs.

## System Installation

Step 0: Ensure a Nvidia GPU is installed in the system and the drivers for the GPU are installed.

Step 1: Create an anaconda environment

Step 2: Activate the environment

Step 3: Execute the requirements.txt file using the command:

```bash
python -m pip install -r requirements.txt
```

## System Usage

The LPDR system executes via the terminal shell. The system is capable of making predictions on image files, video files and live video feeds. The commands to execute the system are shown below

### Images

```bash
python main.py -i <path to image>
```

OR

```bash
python main.py --image <path to image file>
```

### Videos

```bash
python main.py -v <path to video file>
```

OR

```bash
python main.py --video <path to video file>
```

### Live Feeds

```bash
python main.py -u <link to video stream>
```

OR

```bash
python main.py --url <link to video stream>
```
