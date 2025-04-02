docker rm -f lerobot_ros
DIR=$(pwd)/
# docker run --gpus all -it --shm-size=1g --privileged --network=host --name lerobot_ros -v "$DIR":"$DIR" -v /home:/home -v /mnt:/mnt -v /media:/media  -v ~/.imitator:/root/.imitator lerobot_ros:latest bash -c "cd $DIR && bash"
docker run --gpus all -it --shm-size=1g --privileged --network=host --name lerobot_ros -v "$DIR":"$DIR" -v /mnt:/mnt -v /media:/media -v ~/.imitator:/home/user/.imitator lerobot_ros:latest bash -c "cd $DIR && bash"
