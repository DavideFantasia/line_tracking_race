# Line Tracking Race ROS2 Jazzy + Gazebo Harmonic Version  (Work in progress)
This repository collects the starting material for the Line Tracking Race project with Gazebo Harmonics and ROS2.

## 1. Install
### 1.1 Preliminaries
```bash
sudo apt install python3-vcstool python3-colcon-common-extensions git wget
```

### 1.2 Clone and Build
Clone in your workspace (e.g., `~/ros2_ws/src`):
```bash
git clone git@github.com:BRAIR-Education/line_tracking_race.git && git checkout jazzy
```
Build the simulation:
```bash
cd ~/ros2_ws
source /opt/ros/${ROS_DISTRO}/setup.bash
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -i -y --rosdistro ${ROS_DISTRO}
colcon build
```

## 2. Usage
To run the simulation with default weights, execute the following command in the terminal:
```bash
ros2 launch line_tracking_race_bringup line_tracking_race.launch.py
```
Alternatively, to run the offline training with synthesized scenarios, use:
```bash
python3 src/line_tracking_race/line_tracking_race_application/optimizer.py --strategy offline
```
### 2.1 Arguments
Inspect the Launch file for the complete list of parameters.

The most relevant ones are:
- `mode` (`run`/`train`): Defines the operating mode of the controller node:
    - _run mode_: Drives the robot using the NMPC based on the provided weights (either default or custom).
    - _train mode_: Runs multiple episodic simulations of a specified duration to evaluate the fitness of different weight combinations. A Genetic Algorithm iteratively optimizes these sets of weights.
    ```bash 
    ros2 launch line_tracking_race_bringup line_tracking_race.launch.py mode:=run
    ```
    ```bash 
    ros2 launch line_tracking_race_bringup line_tracking_race.launch.py mode:=train
    ```
- `eval_duration` (`float`): Used in _train mode_ to specify the duration (in seconds) of each training episode. The default value is `30.0`.
    ```bash 
    ros2 launch line_tracking_race_bringup line_tracking_race.launch.py mode:=train eval_duration:=30.0
    ```
- `weights` (`[w_d, w_psi, w_effort, k_curve, w_v, qf_mult_d, qf_mult_psi, gamma_decay_start, horizon]`): Sets custom weights for the NMPC when operating in _run mode_. This array also serves as the genome optimized by the Genetic Algorithm.
Semantics:

  **Semantics**:
    - _w_d_: Weight for the lateral position error (cross-track error).
    - _w_psi_: Weight for the orientation (heading) error.
    - _w_effort_: Penalty for the steering effort.
    - _k_curve_:  Braking aggressiveness constant used when approaching curves.
    - _w_v_: Penalty for velocity deviation.
    - _qf_mult_d_:  Terminal cost multiplier for the lateral error.
    - _qf_mult_psi_: Terminal cost multiplier for the orientation error.
    - _gamma_decay_start_: Distance (in meters) to trust the estimated polynomial curvature before it linearly decays to zero.
    - _horizon_: Number of predictive steps evaluated in the NMPC horizon.
    Is is used as the genome that the Genetic Algorithm needs to learn
    ```bash 
    ros2 launch line_tracking_race_bringup line_tracking_race.launch.py weights:='[7.0, 1.0, 0.5, 15.0, 5.0, 1.0, 3.0, 1.0, 30.0]' 
    ```
- `use_ros2_control` (`bool`): Use ros2_control [differential drive controller](https://control.ros.org/jazzy/doc/ros2_controllers/diff_drive_controller/doc/userdoc.html) instead of Gazebo's [DiffDrive](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1DiffDrive.html).
    ```bash 
    ros2 launch line_tracking_race_bringup line_tracking_race.launch.py use_ros2_control:=true
    ```
- `rviz` (`bool`): Start Rviz2
    ```bash 
    ros2 launch line_tracking_race_bringup line_tracking_race.launch.py rviz:=true
    ```



## 3. Topics and Services
Explore using the ROS2 command line which information are available and design your high-level node. You can use the package `line_tracking_race_application` as container of your nodes.
