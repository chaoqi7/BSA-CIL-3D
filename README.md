# BSA-CIL-3D
Boosting the Class-Incremental Learning in 3D Point Clouds via Zero-Collection-Cost Basic Shape Pre-Training.

## 📖Content
- [BSA Dataset](#BSA-Dataset)
- [Pretrained Models](#Pretraining-Models)
- [Code](#Code)

## 🎨BSA Dataset
- The Dataset Creation Process
<p align="center"><img align="center" width="920" src="./BSA_Dataset.png"/></p>

- [Data Samples](./BSA_Dataset)

- [Dataset Generation Code](./BSA_Generation.py)
  
## 🌈Pretrained Models
The pre-trained models are available [[LINK](https://www.alipan.com/s/Jr3T2QMi6Cf)].
- The dVAE model is embedded in the tokenizer to supervise the predicted tokens in the pre-training stage.
- The Point-bert model is embedded in the backbone for continual learning.

## 💻Code
