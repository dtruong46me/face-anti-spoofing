
```
    ______                   ___          __  _    _____                   _____
   / ____/___ _________     /   |  ____  / /_(_)  / ___/____  ____  ____  / __(_)___  ____ _
  / /_  / __ `/ ___/ _ \   / /| | / __ \/ __/ /   \__ \/ __ \/ __ \/ __ \/ /_/ / __ \/ __ `/
 / __/ / /_/ / /__/  __/  / ___ |/ / / / /_/ /   ___/ / /_/ / /_/ / /_/ / __/ / / / / /_/ /
/_/    \__,_/\___/\___/  /_/  |_/_/ /_/\__/_/   /____/ .___/\____/\____/_/ /_/_/ /_/\__, /
                                                    /_/                            /____/
```

## Problem Description

This project aims to evaluate the performance of lightweight face models for facial anti-spoofing (FAS), comparing their accuracy and computational complexity with state-of-the-art deep models.

**Input:** Image with face

**Output:** Fake or Real

## Dataset

LCCD FASD Dataset. Link: https://www.kaggle.com/datasets/faber24/lcc-fasd

## Methods
- [x] ResNeXT50

- [x] MobileNetV3

- [x] FeatherNet

## Results
We evaluate on LCCD FASD Development after preprocessing. Link: [CV_dataset](https://www.kaggle.com/datasets/valleyy/cv-dataset)
|Model|APCER|NPCER|ACER|
|-|:-:|:-:|:-:|
|ResNeXT50|0.1308|0.3037|0.2127|
|MobileNetV3|0.1727|0.2111|0.1917|
|FeatherNet|0.1994|0.1284|0.1639|

## Installation

1. Clone Project
```
git clone https://github.com/dtruong46me/face-anti-spoofing.git
cd face-anti-spoofing
```

2. Install requirements.txt
```
bash setup.sh
```

3. Training
First, you should download [CV_dataset](https://www.kaggle.com/datasets/valleyy/cv-dataset) and put it into the folder `/face-anti-spoofing/cv-dataset` and run the scipt:

```
python run_training.py \
--train_path "cv-dataset/final_data/train" \
--test_path "cv-dataset/final_data/valid" \
--batch_size 128 \
--modelname "seresnext50"\
--wandb_token "<your_wandb_token>" \
--wandb_runname "<your_wandb_run_name>" \
--num_classes 2 \
--max_epochs 40
```
or your can use `bash train_all.sh` to train, evaluate, predict all pretrained models.

You can follow scipts in the notebook: https://www.kaggle.com/code/dtruon46/master-face-anti-spoofing or the file: `/face-anti-spoofing/code train.ipynb` 

4. Demo

You can download weights of models (`.ckpt` file) and put it into the `/face-anti-spoofing/FAS_detector/model/your_model.ckpt`. Put your test image in the `/face-anti-spoofing/assets/samples/your_images.jpg`

- ResNeXT50: [Weights](https://www.kaggle.com/models/dtruon46/resnext50-face-anti-spoofing) (106Mb)
- MobileNetV3: [Weights](https://www.kaggle.com/models/dtruon46/mobilenetv3-face-anti-spoofing) (28MB)
- FeatherNet: [Weights](https://www.kaggle.com/models/dtruon46/feathernet-face-anti-spoofing) (4.3MB)

Then run the scripts

```
python predict_sample.py \
--model_checkpoint "your_model.ckpt" \
--image "your_test_image.jpg"\
--modelname "seresnext50"
```


## Contributions
- Supervisor: Prof. Dang Tuan Linh

- Group Members

|No.|Name|Student ID|Email|
|:-:|-|:-:|-|
|1|Vu Tuan Minh (Leader)|20210597|minh.vt210597@sis.hust.edu.vn|
|2|Nguyen Tien Doanh|20214881|doanh.nt214881@sis.hust.edu.vn|
|3|Nguyen Tung Luong|20214913|luong.nt214913@sis.hust.edu.vn|
|4|Phan Dinh Truong|20214937|truong.pd214937@sis.hust.edu.vn|
