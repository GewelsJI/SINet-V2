# 2021-TPAMI-Concealed Object Detection (SINetV2-Jittor Implementation)

## Introduction

The repo provides inference code of **SINet-V2 (TPAMI-2021)** with [Jittor deep-learning framework](https://github.com/Jittor/jittor).

> **Jittor** is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. The front-end language is Python. Module Design and Dynamic Graph Execution is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA, C++.

## Usage

SINet-V2 is also implemented in the Jittor toolbox which can be found in `./jt_lib`.
+ Create environment by `python3.7 -m pip install jittor` on Linux. 
As for MacOS or Windows users, using Docker `docker run --name jittor -v $PATH_TO_PROJECT:/home/SINet-V2 -it jittor/jittor /bin/bash` 
is easier and necessary. 
A simple way to debug and run the script is running a new command in the container through `docker exec -it jittor /bin/bash` and start the experiments. (More details refer to this [installation tutorial](https://github.com/Jittor/jittor#install))

+ First, run `sudo sysctl vm.overcommit_memory=1` to set the memory allocation policy.

+ Second, switch to the project root by `cd /home/SINet-V2`

+ For testing, run `python3.7 jittor_lib/MyTesting.py`. 

> Note that the Jittor model is just converted from the original PyTorch model via toolbox, and thus, the trained weights of PyTorch model can be used to the inference of Jittor model.

## Performance Comparison

The performance has slight difference due to the different operator implemented between two frameworks.  The download link ([Pytorch](https://drive.google.com/file/d/1I3vKdcjafkTb2U2pOke07khurXxqLpzR/view?usp=sharing) / [Jittor](https://drive.google.com/file/d/13DeX-IMFE6u0TnNG5blUvHzo5o21cVpc/view?usp=sharing)) of prediction results on four testing dataset, including CHAMELEON, CAMO, COD10K, NC4K.

| CHAMELEON dataset    	| $S_\alpha$  	| $E_\phi$  	| $F_\beta^w$  	| M     	|
|----------------------	|-------------	|-----------	|--------------	|-------	|
| PyTorch              	| 0.888       	| 0.942     	| 0.816        	| 0.030 	|
| Jittor               	| 0.890      	| 0.943       	| 0.819        	| 0.030    	|

|  CAMO-Test dataset   	| $S_\alpha$  	| $E_\phi$  	| $F_\beta^w$  	| M     	|
|----------------------	|-------------	|-----------	|--------------	|-------	|
|  PyTorch             	| 0.820       	| 0.882     	| 0.743        	| 0.070 	|
|  Jittor              	| 0.820       	| 0.881     	| 0.743        	| 0.070 	|

|  COD10K-Test dataset 	| $S_\alpha$  	| $E_\phi$  	| $F_\beta^w$  	| M     	|
|----------------------	|-------------	|-----------	|--------------	|-------	|
|  PyTorch             	| 0.815       	| 0.887     	| 0.680        	| 0.037 	|
|  Jittor              	| 0.816       	| 0.888     	| 0.681        	| 0.037 	|

## Speedup

The jittor-based code can speed up the inference efficiency.

| Batch Size  	|     PyTorch    	|     Jittor     	|     Speedup    	|
|-----------	|----------------	|----------------	|----------------	|
|     1     	|     52 FPS     	|     70 FPS     	|     1.35x       	|
|     4     	|     181 FPS    	|     275 FPS    	|     1.52x       	|
|     8     	|     372 FPS    	|     509 FPS    	|     1.37x      	|
|     16    	|     466 FPS    	|     577 FPS    	|     1.24x       	|

## Citation

If you find our work useful in your research, please consider citing:
    
    
    @article{fan2021concealed,
      title={Concealed Object Detection},
      author={Fan, Deng-Ping and Ji, Ge-Peng and Cheng, Ming-Ming and Shao, Ling},
      journal={IEEE TPAMI},
      year={2021}
    }
    
    @article{hu2020jittor,
      title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
      author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
      journal={Information Sciences},
      volume={63},
      number={222103},
      pages={1--21},
      year={2020}
    }
