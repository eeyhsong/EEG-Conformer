# EEG-Conformer

### EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization [[Paper](https://ieeexplore.ieee.org)]

## Abstract
![Network Architecture](Fig1.png)

Due to the limited perceptual field, convolutional neural networks (CNN) only extract local temporal features and may fail to capture long-term dependencies for EEG decoding.
In this paper, we propose a compact convolutional Transformer, named EEG Conformer, to encapsulate local and global features in a unified EEG classification framework. 
Specifically, the convolution module learns the low-level local features throughout the one-dimensional temporal and spatial convolution layers.  
The self-attention module is straightforwardly connected to extract the global correlation within the local temporal features. 
Subsequently, the simple classifier module based on fully-connected layers is followed to predict the categories for EEG signals.
To enhance the interpretability, we also devise a visualization strategy to project the class activation mapping onto the brain topography. 
Finally, we have conducted extensive experiments to evaluate our method on three public datasets in EEG-based motor imagery and emotion recognition paradigms. 
The experimental results show that our method achieves state-of-the-art performance and has great potential to be a new baseline for general EEG decoding. 


## Requirmenets:
- Python 3.10
- Pytorch 1.12


## Datasets
- [BCI_competition_IV2a](https://www.bbci.de/competition/iv/)
- [BCI_competition_IV2b](https://www.bbci.de/competition/iv/)
- [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html)


## Citation
Hope this code can be useful. I would be very appreciate if you cite us in your paper. 😄
```
@article{song2021eegconformer,
  author={Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization}, 
  year={2023},
  volume={},
  pages={},
  doi={}
}
``` 
