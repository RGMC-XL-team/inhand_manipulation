# Task A

## Installation

### Simulation

Install ROS Noetic.

In Python==3.8.10 env:

```bash
pip install mujoco
pip install git+https://github.com/google-deepmind/dm_control.git # install the latest dm_control
pip install rospkg
pip install pandas
pip install scikit-learn
```

Install pinocchio in your python env [doc](https://github.com/conda-forge/pinocchio-feedstock#installing-pinocchio) .

Build the ROS workspace:

```
cd <your_ros_workspace>
catkin_make
```

## Usage

### Simulation

Run the mujoco simulation only (no motion):

```bash
# in python env
cd leap_task_A/scripts
source ../../../../devel/setup.bash
python leaphand_mujoco.py
```

Run the in-hand object moving:

```bash
# in python env
cd leap_task_A/scripts
source ../../../../devel/setup.bash
python leaphand_control.py
```
