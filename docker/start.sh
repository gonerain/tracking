#!/bin/bash
docker container ls -a -f name=rospytorch | grep rospytorch$ > /dev/null
if [ $? == 0 ]
then
	docker container start rospytorch
	docker exec -it rospytorch /bin/bash -c 'source /opt/ros/noetic/setup.bash && exec /bin/bash'
else
	xhost +
	docker run --gpus all -it -v $PWD/../:/root/HDMap -v /tmp/.X11-unix:/tmp/.X11-unix -e -e DISPLAY=$DISPLAY -e GDK_SCALE=$GDK_SCALE -e GDK_DPI_SCALE=$GDK_DPI_SCALE  --name rospytorch ebhrz/ros-pytorch:noetic_pt110_cu113 /bin/bash -c 'source /opt/ros/noetic/setup.bash && exec /bin/bash'
fi

