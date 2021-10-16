# Do you see what I see? 
Code related to Eye-gaze based analysis on VQA

## Dataset Columns

|Column Name|Data Type |Description|
|-----------|----------|-----------|
|dummy1|int64|question_id|
|dummy2|str|question|


## Available options and usage

### 1. Setup and download datasets
#### Usage
```bash
sh setup.sh
```
Downloads and unpacks necessary datasets into `gaze_vqa_analysis/data/`

https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ

https://drive.google.com/file/d/1iDsjZmNVrkGG-21HqpePZMhWOrraet__/view
### 2. Gaze to image bounding box mapping and plotting

#### Usage
```python
python plot_quantized_gaze.py
```
#### Arguments and Description
|Argument |Data Type |Default |Description|
|---------|----------|--------|-----------|
|dummy1|int64|0|dummy|

### 3. Image attention plotting for MCAN
### Usage
```python
python scripts/plot_attn.py
```





