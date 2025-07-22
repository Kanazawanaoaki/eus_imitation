# lerobot_ros

# Setup
In server PC
```bash
cd lerobot_ros
docker build -t lerobot_ros .
```

# Usage

## Data Conversion
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
cd lerobot_ros/scripts
python build_data_from_rosbag.py -pn [project_name] -d ~/.imitator/[project_name]/data/rosbags
```

Convert pkl to lerobot datasets.
```bash
cd lerobot_ros/scripts
python convert_data_to_lerobot.py -pn [project_name]
```

## Learning Policy
Learning Diffusion Policy.
```bash
## in Azure
python -m lerobot.scripts.train --output_dir=outputs/train/[output_name] --policy.type=diffusion --policy.push_to_hub=false --policy.use_separate_rgb_encoder_per_camera=true --dataset.repo_id=my_data/[project_name] --dataset.root=/mount/kanazawaa100/kanazawafiles/lerobot_home/my_data/[data_name] --job_name=diffusion_[project_name] --resume=false --num_workers=4 --batch_size=64 --steps=100000 --log_freq=200 --wandb.enable=true --wandb.disable_artifact=true --wandb.project=lerobot

## in TR desktop PC
python -m lerobot.scripts.train --output_dir=outputs/train/[output_name] --policy.type=diffusion --policy.push_to_hub=false --policy.use_separate_rgb_encoder_per_camera=true --dataset.repo_id=my_data/[project_name] --dataset.root=/media/ptolemy/73B2/kanazawa/datas/lerobot_home/my_data/[data_name] --job_name=diffusion_[project_name] --resume=false --num_workers=4 --batch_size=64 --steps=100000 --log_freq=200 --wandb.enable=true --wandb.disable_artifact=true --wandb.project=lerobot
```

Learning ACT Policy.
```bash
## in Azure
python -m lerobot.scripts.train --output_dir=outputs/train/[output_name] --policy.type=act --policy.push_to_hub=false --dataset.repo_id=my_data/[project_name] --dataset.root=/mount/kanazawaa100/kanazawafiles/lerobot_home/my_data/[data_name] --job_name=act_[project_name] --resume=false --num_workers=4 --batch_size=64 --steps=100000 --log_freq=200 --wandb.enable=true --wandb.disable_artifact=true --wandb.project=lerobot

## in TR desktop PC
python -m lerobot.scripts.train --output_dir=outputs/train/[output_name] --policy.type=act --policy.push_to_hub=false --dataset.repo_id=my_data/[project_name] --dataset.root=/media/ptolemy/73B2/kanazawa/datas/lerobot_home/my_data/[data_name] --job_name=act_[project_name] --resume=false --num_workers=4 --batch_size=64 --steps=100000 --log_freq=200 --wandb.enable=true --wandb.disable_artifact=true --wandb.project=lerobot
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

Specify the model path and execute the policy.
```bash
cd node_scripts
python arg_path_ros_diffusion_policy_executor.py -pn [project_name] -hz [ros node hz] -m [model path]
```