# SINet-V2 with Jittor framework

## Introduction

The repo provides inference code of SINet-V2 (TPAMI-2021) with Jittor deep-learning framework.

## Usage

SINet-V2 is also implemented in the Jittor toolbox which can be found in `./jt_lib`.
+ Create environment by `python3.7 -m pip install jittor` on Linux. 
As for MacOS or Windows users, using Docker `docker run --name jittor -v $PATH_TO_PROJECT:/home/SINet-V2 -it jittor/jittor /bin/bash` 
is easier and necessary. 
A simple way to debug and run the script is running a new command in the container through `docker exec -it jittor /bin/bash` and start the experiments.

+ First, run `sudo sysctl vm.overcommit_memory=1` to set the memory allocation policy.

+ Second, switch to the project root by `cd /home/SINet-V2`

+ For testing, run `python3.7 jittor/MyTesting.py`. 

## Performance

The performance has slight difference due to the difference of two frameworks. Note that the Jittor model is just converted from the original PyTorch model via toolbox, and thus, the trained weights of PyTorch model 
can be used to the inference of Jittor model.

| CHAMELEON dataset    	| $S_\alpha$  	| $E_\phi$  	| $F_\beta^w$  	| M     	|
|----------------------	|-------------	|-----------	|--------------	|-------	|
| PyTorch              	| 0.888       	| 0.942     	| 0.816        	| 0.030 	|
| Jittor               	|             	|           	|              	|       	|

|  CAMO-Test dataset   	| $S_\alpha$  	| $E_\phi$  	| $F_\beta^w$  	| M     	|
|----------------------	|-------------	|-----------	|--------------	|-------	|
|  PyTorch             	| 0.820       	| 0.882     	| 0.743        	| 0.070 	|
|  Jittor              	|             	|           	|              	|       	|

|  COD10K-Test dataset 	| $S_\alpha$  	| $E_\phi$  	| $F_\beta^w$  	| M     	|
|----------------------	|-------------	|-----------	|--------------	|-------	|
|  PyTorch             	| 0.815       	| 0.887     	| 0.680        	| 0.037 	|
|  Jittor              	|             	|           	|              	|       	|