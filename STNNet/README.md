# Space-Time Neighbor-Aware Network (STNNet-pytorch)

This is the PyTorch version repository for [Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark](https://arxiv.org/abs/1912.01811).

## Prerequisites
We use Anaconda as the environment. The code was tested on Ubuntu 18.04, with Python 3.6, CUDA 10.2 and PyTorch v1.6.0. NVIDIA GPUs are needed for both training and testing.

Install PyTorch==1.6 and torchvision==0.7:
```
conda create -n STTNet python=3.6 pytorch=1.6 torchvision -c pytorch
```

Install package dependencies:
```
pip install -r requirements.txt
```

## Datasets
Download the [DroneCrowd](https://drive.google.com/drive/folders/1EUKLJ1WmrhWTNGt4wFLyHRfspJAt56WN?usp=sharing) data, and then unzip them and rename them under the directory like:

```
dataset
├── train_data
│   └── ground_truth ..
│   └── images ..
├── val_data
│   └── ground_truth ..
│   └── images ..
├── test_data
│   └── ground_truth ..
│   └── images ..
├── annotations
├── make_data_density.py
└── make_data_localization.py
```

## Ground-Truth Generation

Please follow the `make_data_density.py` and `make_data_localization.py` to generate the ground-truth of density map and location map respectively.

## Training Process
If only using the density head, run:

```
python mytrain.py den --mGPUs --bs 4
``` 

If using both the density head and the localization head, run:

```
python mytrain.py loc --mGPUs --loc --bs 4
``` 

If removing the relation contraint, run:

```
python mytrain.py tem --mGPUs --loc --trk --tem --bs 4 --pre loc_best_model.pth.tar
``` 

If removing the cycle loss, run:

```
python mytrain.py trk --mGPUs --loc --trk --bs 4 --pre loc_best_model.pth.tar
``` 

For STNNet, run:

```
python mytrain.py cyc --mGPUs --trk --cyc --bs 4 --pre loc_best_model.pth.tar
``` 

The pre-trained models are downloaded from [here](https://drive.google.com/file/d/1H-GrqyYxERLzVM5wQblUDeX24Ht2lD6q/view?usp=sharing).

## Testing Process
If evaluating the STNNet variant with the density head, run:

```
python mytest.py
``` 

If evaluating the STNNet variant with both the density head and the localization head, run:

```
python mytest.py --loc
``` 

If evaluating the STNNet variant without relation constraint, run:

```
python mytest.py --loc --trk --tem
``` 

If evaluating the STNNet variant without cycle loss, run:

```
python mytest.py --loc --trk
``` 

If evaluating the STNNet, run:

```
python mytest.py --loc --trk --cyc
``` 

To evaluate detection and tracking performance, use the `DroneCrowd-VID-toolkit` and `DroneCrowd-MOT-toolkit` respectively. The detection and tracking results are downloaded from [here](https://drive.google.com/file/d/1Ipsjlxk_U3j2cYgQ89o2DgN4wYdDZ8xr/view?usp=sharing).


## References

If you use the STNNet method or the DroneCrowd dataset, please cite our papers. Thank you!

```
@inproceedings{dronecrowd_cvpr2021,
  author    = {Longyin Wen and
               Dawei Du and
               Pengfei Zhu and
               Qinghua Hu and
               Qilong Wang and
               Liefeng Bo and
               Siwei Lyu},
  title     = {Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark},
  booktitle = {CVPR},
  year      = {2021}
}
```

```
@inproceedings{dronecrowd_eccvw2020,
  author    = {Dawei Du and
               Longyin Wen and
               Pengfei Zhu and
               Heng Fan and
               Qinghua Hu and
               Haibin Ling and
               Mubarak Shah and
               Junwen Pan},
  title     = {VisDrone-CC2020: The Vision Meets Drone Crowd Counting Challenge Results},
  booktitle = {ECCVW},
  volume    = {12538},
  pages     = {675--691},
  year      = {2020}
}
```

```
@article{dronecrowd_arxiv2019,
  author    = {Longyin Wen and
               Dawei Du and
               Pengfei Zhu and
               Qinghua Hu and
               Qilong Wang and
               Liefeng Bo and
               Siwei Lyu},
  title     = {Drone-based Joint Density Map Estimation, Localization and Tracking with Space-Time Multi-Scale Attention Network},
  journal   = {CoRR},
  volume    = {abs/1912.01811},
  year      = {2019}
}
```