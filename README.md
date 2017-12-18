# Deep Networks with Stochastic Depth (ECBM4040)
This is a TensorFlow implementation of the following paper: https://arxiv.org/abs/1603.09382.
The code can reproduce the results of the paper on the CIFAR-10 dataset. With very small modifications to the script that downloads data, and changing the output dimensions to 100 instead of 10, the code can run on CIFAR-100 as well.

# Implementation
The code has been tested on a Tesla-K80 GPU, where one epoch takes ~200 seconds for P=0.2 to ~240 seconds for P=1 i.e. when no layer is dropped. It requires the latest version of TensorFlow i.e. `TensorFlow==1.4`

# Getting Started on CIFAR-10
```bash
git clone https://github.com/anujk3/Stochastic_Depth_ECBM4040.git
cd Stochastic_Depth_ECBM4040
python main.py
```
The model can be run on the CIFAR-10 out of the box. It will checkpoint the model at every 10th epoch. Additionally, it will write the training loss and test error at the end of each epoch to TensorBoard which can be visualized using the following code:
```bash
cd Stochastic_Depth_ECBM4040
tensorboard --logdir="tf_logs/"
```
