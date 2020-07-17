# Space-Time Multi-Scale Attention Network (STANet-pytorch)

This is the PyTorch version repository for [Drone-based Joint Density Map Estimation, Localization and Tracking with Space-Time Multi-Scale Attention Network](https://arxiv.org/abs/1912.01811).

## Prerequisites
We use Virtualenv as the environment. The code was tested on Ubuntu 16.04, with Python 2.7, CUDA 9.0 and PyTorch v0.4.0/v0.4.1. NVIDIA GPUs are needed for both training and testing.

Install the requirements:
```
pip install -r requirements.txt
```
## Datasets
[DroneCrowd](https://github.com/VisDrone/Awesome-VisDrone)

## Ground-Truth Generation

Please follow the `make_data.py` and `make_data_max.py` to generate the ground truth of density map and location map respectively.

## Training Process
If only using the density head, run:
```
python mytrain.py models/den --mGPUs --bs 6
``` 
If using both the density head and the localization head, run:
```
python mytrain.py models/loc --mGPUs --loc --bs 6
``` 
For STANet, run:
```
python mytrain.py models/trk --mGPUs --trk --bs 6
``` 
## Testing Process
If evaluating the STANet variant with the density head, run:
```
python mytest.py
``` 
If evaluating the STANet variant with both the density head and the localization head, run:
```
python mytest.py --loc
``` 
If evaluating the STANet, run:
```
python mytest.py --loc --trk
``` 

## References

If you use the results of STANet, please cite our paper. Thank you!

```
@article{DBLP:journals/corr/abs-1912-01811,
  author    = {Longyin Wen and
               Dawei Du and
               Pengfei Zhu and
               Qinghua Hu and
               Qilong Wang and
               Liefeng Bo and
               Siwei Lyu},
  title     = {Drone-based Joint Density Map Estimation, Localization and Tracking
               with Space-Time Multi-Scale Attention Network},
  journal   = {CoRR},
  volume    = {abs/1912.01811},
  year      = {2019}
}
```