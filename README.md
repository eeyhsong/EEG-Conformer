# EEG-Conformer

### EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization [[Paper](https://ieeexplore.ieee.org/document/9991178)]
##### Core idea: spatial-temporal conv + pooling + self-attention

### News
#### 🎉🎉🎉 We've joined in [braindecode](https://braindecode.org/stable/index.html) toolbox. Use [**here**](https://braindecode.org/stable/generated/braindecode.models.EEGConformer.html) for detailed info.


Thanks to [Bru](https://github.com/bruAristimunha) and colleagues for helping with the modifications.

## Abstract
![Network Architecture](/visualization/Fig1.png)

- We propose a compact convolutional Transformer, EEG Conformer, to encapsulate local and global features in a unified EEG classification framework.  
- The convolution module learns the low-level local features throughout the one-dimensional temporal and spatial convolution layers. The self-attention module is straightforwardly connected to extract the global correlation within the local temporal features. Subsequently, the simple classifier module based on fully-connected layers is followed to predict the categories for EEG signals. 
- We also devise a visualization strategy to project the class activation mapping onto the brain topography.


## Requirements:
- Python 3.10
- Pytorch 1.12


## Datasets
Please use consistent train-val-test split when comparing with other methods.
- [BCI_competition_IV2a](https://www.bbci.de/competition/iv/) - acc 78.66% (hold out)
- [BCI_competition_IV2b](https://www.bbci.de/competition/iv/) - acc 84.63% (hold out)
- [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html) - acc 95.30% (5-fold)


## Citation
Hope this code can be useful. I would appreciate you citing us in your paper. 😊
```
@article{song2023eeg,
  title = {{{EEG Conformer}}: {{Convolutional Transformer}} for {{EEG Decoding}} and {{Visualization}}},
  shorttitle = {{{EEG Conformer}}},
  author = {Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
  year = {2023},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume = {31},
  pages = {710--719},
  issn = {1558-0210},
  doi = {10.1109/TNSRE.2022.3230250}
}
``` 

