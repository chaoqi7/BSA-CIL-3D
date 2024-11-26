# BSA-CIL-3D
Boosting the Class-Incremental Learning in 3D Point Clouds via Zero-Collection-Cost Basic Shape Pre-Training.

## 📖Content
- [BSA Dataset](#BSA-Dataset)
- [Pretrained Models](#Pretraining-Models)
- [Code](#Code)

## 🎨BSA Dataset
- The Dataset Creation Process
 
![screenshot](https://cdn.z.wiki/autoupload/20241126/8crj/1345X976/BSA-Dataset-fuben.png)

- [Data Samples](./BSA_Dataset)

- [Dataset Generation Code](./BSA_Generation.py)
  
## 🌈Pretrained Models
The pre-trained models are available [[LINK](https://www.alipan.com/s/Jr3T2QMi6Cf)] (CODE: 7u5q).
- The dVAE model is embedded in the tokenizer to supervise the predicted tokens in the pre-training stage.
- The Point-bert model is embedded in the backbone for continual learning.

## 💻Code
