# SINet-V2 with Jittor framework

SINet-V2 is also implemented in the Jittor toolbox which can be found in `./jt_lib`.
+ Create environment by `python3.7 -m pip install jittor` on Linux. 
As for MacOS or Windows users, using Docker `docker run --name jittor -v $PATH_TO_PROJECT:/home/SINet-V2 -it jittor/jittor /bin/bash` 
is easier and necessary. 
A simple way to debug and run the script is running a new command in the container through `docker exec -it jittor /bin/bash` and start the experiments.

+ First, run `sudo sysctl vm.overcommit_memory=1` to set the memory allocation policy.

+ Second, switch to the project root by `cd /home/SINet-V2`

+ For training, run `python3.7 jittor/MyTrain_Val.py`.

+ For testing, run `python3.7 jittor/MyTesting.py`.