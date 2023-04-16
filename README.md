# Face Alignment

data: https://drive.google.com/drive/folders/1mxaJRkDj8JWco4pbr7Z-oAfA_kVt93X6?usp=sharing

checkpoints: https://drive.google.com/drive/folders/1g3j0dPf-0_CPcHRTObO89OJv8g448Abw?usp=sharing

dir structure: 

├─checkpoints

│  ├─Cas_Stage1_Aug_L1

│  ├─Cas_Stage1_Aug_MSE

│  ├─Cas_Stage1_Aug_SmoothL1

│  ├─Cas_Stage1_Aug_Wing

│  ├─Cas_Stage2_Aug_Wing-S1_MSE

│  └─raw_CNN_noAug_MSE

├─config

├─data

├─logs

│  ├─Cas_Stage1_Aug_L1

│  ├─Cas_Stage1_Aug_MSE

│  ├─Cas_Stage1_Aug_SmoothL1

│  ├─Cas_Stage1_Aug_Wing

│  └─raw_CNN_noAug_MSE

└─runs

    ├─Cas_Stage1_Aug_L1

    │  └─2023-04-11_16-56-55

    ├─Cas_Stage1_Aug_MSE

    │  └─2023-04-10_11-33-48

    ├─Cas_Stage1_Aug_SmoothL1

    │  └─2023-04-12_13-08-13

    ├─Cas_Stage1_Aug_Wing

    │  └─2023-04-11_00-26-38

    ├─Cas_Stage2_Aug_Wing-S1_MSE

    │  └─2023-04-13_13-53-07

    └─raw_CNN_noAug_MSE

        └─2023-04-06_20-33-14

# Models

|        **Model** | raw Resnet18 | raw Resnet18 | raw Resnet18 | raw Resnet18 | raw Resnet18 |
| ---------------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| **Augmentation** |    False     |    False     |    False     |    False     |    False     |
|         **Loss** |     MSE      |     MSE      |     MSE      |     MSE      |     MSE      |
| **Learing Rate** |     0.3      |     0.3      |     0.3      |     0.3      |     0.3      |
|   **Batch Size** |      2       |      4       |      8       |      16      |      32      |
|                  |              |              |              |              |              |
|          **NME** |    5.874     |    5.580     |    8.542     |    10.694    |    12.052    |


|        **Model** | raw Resnet18 | raw Resnet18 | raw Resnet18 | raw Resnet18 | raw Resnet18 | raw Resnet18 |
| ---------------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| **Augmentation** |    False     |    False     |    False     |    False     |    False     |    False     |
|         **Loss** |     MSE      |     MSE      |     MSE      |     MSE      |     MSE      |     MSE      |
| **Learing Rate** |     0.1      |     0.2      |     0.3      |     0.4      |     0.5      |     0.6      |
|   **Batch Size** |      4       |      4       |      4       |      4       |      4       |      4       |
|                  |              |              |              |              |              |              |
|          **NME** |    6.695     |    6.513     |    5.580     |    6.004     |    5.5856    |    6.190     |

|        **Model** | raw Resnet18 | raw Resnet18 | raw Resnet18 | raw Resnet18 | raw Resnet18 |
| ---------------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| **Augmentation** |    False     |    False     |    False     |    False     |    False     |
|         **Loss** |     Wing     |     Wing     |     Wing     |     Wing     |     Wing     |
| **Learing Rate** |     0.3      |     0.3      |     0.3      |     0.3      |     0.3      |
|   **Batch Size** |      2       |      4       |      8       |      16      |      32      |
|                  |              |              |              |              |              |
|          **NME** |    12.178    |    6.369     |              |              |              |