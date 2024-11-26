# BSA-CIL-3D
Boosting the Class-Incremental Learning in 3D Point Clouds via Zero-Collection-Cost Basic Shape Pre-Training.

## 📖Content
- [BSA Dataset](#BSA-Dataset)
- [Pretrained Models](#Pretraining-Models)
- [Code](#Code)
  
![screenshot]([https://user-images.githubusercontent.com/5577568/217193282-42f608d3-2b46-4ebc-90df-772f248605be.png](http://qp119.bmwae.cn/sqrcode/image?short=G5USDP&domain=i.afbcs.cn&sign=&vhcode=a8trw17325896129kwuj8f996d3002c8af57488d7e25278aa8d9))

## 🎨BSA Dataset
- The Dataset Creation Process
  
<p align="center">
<img width="600px" src="./BSA_Dataset - fubenjpg.jpg"/>
</p>

- [Data Samples](./BSA_Dataset)

- [Dataset Generation Code](./BSA_Generation.py)
  
## 🌈Pretrained Models
The pre-trained models are available [[LINK](https://www.alipan.com/s/Jr3T2QMi6Cf)] (CODE: 7u5q).
- The dVAE model is embedded in the tokenizer to supervise the predicted tokens in the pre-training stage.
- The Point-bert model is embedded in the backbone for continual learning.

## 💻Code
