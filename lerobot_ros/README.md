# lerobot_ros

# Setup
In server PC
```bash
cd lerobot_ros
docker build -t lerobot_ros .
```

# Usage

## Learning Policy
Launch container (The following operations are basically run within the container).
```bash
cd lerobot_ros
bash run_container.sh
```
Create a project directory under ~/.imitator.
```bash
imitator init [project_name]
```
Edit `~/.imitator/[project_name]/config/config.yaml` with [this file](https://github.com/Kanazawanaoaki/eus_imitation/blob/add-diffusion-policy/lerobot_ros/examples/pr2_imitator_sample_config.yaml) as a reference.

Create a symbolic link to the rosbag data folder.
```bash
ln -sf [path to your rosbag dir] ~/.imitator/[project_name]/data/rosbags
```

Convert rosbag data to pkl file.
```bash
cd scripts
python build_data_from_rosbag.py -pn [project_name] -d ~/.imitator/[project_name]/data/rosbags
```

Learning Policy.
```bash
cd lerobot_ros/scripts
python build_data_rosbag_episode.py -pn [project_name]
```

## Execute Policy
Launch container (The following operations are basically run within the container).
```bash
cd lerobot_ros
bash run_container.sh
```
execute diffusion policy with ros.
```bash
cd node_scripts
python ros_diffusion_policy_executor.py -pn [project_name] -hz [ros node hz]
```