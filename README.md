# Face Alignment
This repository contains the code for the assignment for Coursework_Assignment_Spring_23.pdf.

## Requirements
- Python >= 3.9
- torch >= 2.0.0
- opencv-python >= 4.7.0.72
- numpy >= 1.23.5
- matplotlib >= 3.7.0
- tensorboard >= 2.12.2
- torchlm >= 0.1.6.10
- tqdm >= 4.65.0
- pyyaml >= 6.0
- torchsummary >= 1.5.1

## File Structure
```
/data
The data is available at:
https://drive.google.com/drive/folders/1mxaJRkDj8JWco4pbr7Z-oAfA_kVt93X6?usp=sharing

/checkpoints
Best checkpoints for each model are available at:
https://drive.google.com/drive/folders/1g3j0dPf-0_CPcHRTObO89OJv8g448Abw?usp=sharing

/configs
The folder of Inference contains the config files for inference, and the left folders contain the config files for training.

/logs
This folder is used to save the logs of training and inference.

/runs
This folder is used to save the tensorboard logs.

/results
Experiment Results: /results/results.xlsx
Predictions of Cascade Model on test set: /results/results.csv
Model flowchart: /results/Model
Augmentation Visualization: /results/Visualization/Augmentation
Visualization of the results of the model on examples: /results/Visualization/Visualization_Examples
Visualization of the results of the model on the test set and train_test set: /results/Visualization/Vizualization_Test

/criteria.py
This file contains the custom loss functions.

/dataset.py
This file contains the custom dataset class and augmentation functions.

inference.py
This file is used to run inference.

main.py
This file is used to run training.

resnet.py
This file contains the definition of the ResNet model.

utils.py
This file contains the tool functions. 

Coursework_Assignment_Spring_23.pdf is the description of the assignment.

affineTransformation.ipynb, draft.py are all dafts for the affine transformation, pls ignore them.
```

## Train

```python train.py --config configs\Cas_Stage2_noAug_MSE-S1_noAug_MSE\Cas_Stage2_noAug_MSE-S1_noAug_MSE-lr0.5g0.9_B2.yaml```

## Inference
```python inference.py --config configs\Inference\Inference_Cas_Stage2_noAug_MSE-S1_noAug_MSE-lr0.5g0.9_B2.yaml```

## Reference
Feng, Z.-H. et al. (2017) ‘Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks’. doi:10.48550/arxiv.1711.06753.

MMPose Contributors. (2020). OpenMMLab Pose Estimation Toolbox and Benchmark (Version v1.0.00). https://github.com/open-mmlab/mmpose

DefTruth, Lalu Erfandi Maula Yusnu. Torch Lm: A PyTorch Landmarks-Only Library (Version v0.1.6.10). https://github.com/DefTruth/torchlm
