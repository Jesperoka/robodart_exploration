

To build docker container
```
docker build -t franka-test .
```


To run docker container
```
docker run -it \
    --net=host \
    --privileged \
    --mount type=bind,source=$(pwd)/robodart_exploration,target=/robodart_exploration_vm \
    franka-test:latest 
```

Other useful options:
```
    --mount type=bind,source=$(pwd)/your-folder-on-host,target=/your-folder-on-vm \
    --device=/dev/ttyUSB0 \
```


To build+run in single command:
```
docker run -it \
    $(docker build -q .)
```