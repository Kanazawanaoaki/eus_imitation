docker rm -f lerobot_ros
DIR=$(pwd)/
docker run --gpus all -it --privileged --network=host --name lerobot_ros -v "$DIR":"$DIR" -v /home:/home -v /mnt:/mnt lerobot_ros:latest bash -c "cd $DIR && bash"
