# DroneCrowd
Paper  [Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark](https://openaccess.thecvf.com/content/CVPR2021/papers/Wen_Detection_Tracking_and_Counting_Meets_Drones_in_Crowds_A_Benchmark_CVPR_2021_paper.pdf).

## Introduction
![VisDrone](https://github.com/VisDrone/DroneCrowd/blob/master/sample.png)

This paper proposes a space-time multi-scale attention network (STANet) to solve density map estimation, localization and tracking in dense crowds of video clips captured by drones with arbitrary crowd density, perspective, and flight altitude. Our STANet method aggregates multi-scale feature maps in sequential frames to exploit the temporal coherency, and then predict the density maps, localize the targets, and associate them in crowds simultaneously. A coarse-to-fine process is designed to gradually apply the attention module on the aggregated multi-scale feature maps to enforce the network to exploit the discriminative space-time features for better performance. The whole network is trained in an end-to-end manner with the multi-task loss, formed by three terms, i.e., the density map loss, localization loss and association loss. The non-maximal suppression followed by the min-cost flow framework is used to generate the trajectories of targets' in scenarios. Since existing crowd counting datasets merely focus on crowd counting in static cameras rather than density map estimation, counting and tracking in crowds on drones, we have collected a new large-scale drone-based dataset, DroneCrowd, formed by 112 video clips with 33,600 high resolution frames (i.e., 1920x1080) captured in 70 different scenarios. With intensive amount of effort, our dataset provides 20,800 people trajectories with 4.8 million head annotations and several video-level attributes in sequences. Extensive experiments are conducted on two challenging public datasets, i.e., Shanghaitech and UCF-QNRF, and our DroneCrowd, to demonstrate that STANet achieves favorable performance against the state-of-the-arts. 

## Dataset

### ECCV2020 Challenge

The VisDrone 2020 Crowd Counting Challenge requires participating algorithms to count persons in each frame. The challenge will provide 112 challenging sequences, including 82 video sequences for training (2,420 frames in total), and 30 sequences for testing (900 frames in total), which are available on the download page. We manually annotate persons with points in each video frame. 

DroneCrowd (1.03 GB): [BaiduYun](https://pan.baidu.com/share/init?surl=llJZJMi2L5oUQvj31iBlfg)(code: h0j8)| [GoogleDrive](https://drive.google.com/file/d/1HY3V4QObrVjzXUxL_J86oxn2bi7FMUgd/view?usp=sharing) 

### DroneCrowd (Full Version)
This full version consists of 112 video clips with 33,600 high resolution frames (i.e., 1920x1080) captured in 70 different scenarios.  With intensive amount of effort, our dataset provides 20,800 people trajectories with 4.8 million head annotations and several video-level attributes in sequences.  

DroneCrowd [BaiduYun](https://pan.baidu.com/s/1hjXoVZJ16y9Tf7UXcJw3oQ)(code:ml1u)| [GoogleDrive](https://drive.google.com/drive/folders/1EUKLJ1WmrhWTNGt4wFLyHRfspJAt56WN?usp=sharing) 

## Code

[Space-Time Neighbor-Aware Network (STNNet-pytorch)](https://github.com/VisDrone/DroneCrowd/tree/master/STNNet)

[Space-Time Multi-Scale Attention Network (STANet-pytorch)](https://github.com/VisDrone/DroneCrowd/tree/master/STANet)


## Citation

Please cite this paper if you want to use it in your work.
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
@article{zhu2021graph,
  title={Graph Regularized Flow Attention Network for Video Animal Counting from Drones},
  author={Zhu, Pengfei and Peng, Tao and Du, Dawei and Yu, Hongtao and Zhang, Libo and Hu, Qinghua},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```
