# ViTexNet: A Multi-Modal Vision-Text Fusion Network for Chest X-Ray Image Segmentation

This is an official PyTorch implementation of "ViTexNet: A Multi-Modal Vision-Text Fusion Network for Chest X-Ray Image Segmentation"

>  
-----

## Framework
![Framework]()

## Installation  
The main dependencies are as follows:  
```
einops
linformer
monai
pandas 
pytorch_lightning
timm
torch
torchmetrics 
torchvision 
transformers 
thop
```
or use the following:
```
pip install requirements.txt
```

## Requirements
### Dataset
1. The images and segmentation masks of **QaTa-COV19** dataset are fetched from this [link](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset).

   **We used QaTa-COV19-v2 in this experiment**.
2. The text annotations of **QaTa-COV19** dataset are taken from this GitHub [repo](https://github.com/HUANGLIZI/LViT).

   **Thanks to Li *et al.* for their contributions. If you use this text annotations, please cite their work**.

### Pre-trained Model Weights
We have used [Swin-Tiny-Patch4-Window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224) (vision) and [BiomedVLP-CXR-BERT-Specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized) (text) in this experiment.

The models can be used as follows:
   ```
   url = "microsoft/swin-tiny-patch4-window7-224"
   tokenizer = AutoTokenizer.from_pretrained(url,trust_remote_code=True)
   model = AutoModel.from_pretrained(url, trust_remote_code=True)
   ```

## Repository Structure
```
ViTexNet
├── data
    ├── test
    |   ├── images
    |   |   ├── image_1.png
    |   |   ├── image_2.png
    |   |   ├── ...
    │   ├── masks
    |   |   ├── mask_image_1.png
    |   |   ├── mask_image_2.png
    |   |   ├── ...
    ├── train
    |   ├── images
    |   |   ├── image_1.png
    |   |   ├── image_2.png
    |   |   ├── ...
    │   ├── masks
    |   |   ├── mask_image_1.png
    |   |   ├── mask_image_2.png
    |   |   ├── ...
    ├── test_annotations.csv
    ├── train_annotations.csv
├── train.py
├── test.py
├── ...
```

## Code Execution
### Train
To **train** the model, execute: ``` python train_net.py ```

### Test
To **test** the model, execute: ``` python test_net.py ``` after *training* the model or using the learned weights given [*below*](#model-weights).

## Model Weights
The learned model weights are available below:
| Dataset | Model | Download link |
| ----------- | ------- | ---------------- |
| QaTa-COV19 | ViTexNet | <div align="center"><a href="https://drive.google.com/">Google Drive</a></div> |

## Todo List
- [ ] Release entire code
- [X] Release model weights

## Acknowledgement
The work is inspired from [LViT](https://github.com/HUANGLIZI/LViT) and [Ariadne’s Thread](https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023). Thanks for the open source contributions!

## Citation
If you find our work useful, please cite our paper:
```

```
