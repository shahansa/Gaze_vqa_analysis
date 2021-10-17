# Do you see what I see? 
Code related to Eye-gaze based analysis on VQA

## Dataset
Download from [shared link](https://drive.google.com/file/d/1iDsjZmNVrkGG-21HqpePZMhWOrraet__/view)

## Dataset Columns

|Column Name|Data Type |Description|
|-----------|----------|-----------|
|QuestionId|int64|Unique question id from VQA V2.0|
|ImageFileName|str|Image file name from VQA V2.0|
|Question|str|Question from VQA V2.0|
|ImageSize_x|int64|Length of image|
|ImageSize_y|float64|Width of image|
|ImageOrigin_x|float64|x-coordinate of origin when image is displayed in the UI|
|ImageOrigin_y|float64|y-coordinate of origin when image is displayed in the UI|
|Fixations|set/object|Collection of fixations from eye-gaze tracker when the question and image are displayed in the UI|



## Available options and usage

### 1. Setup and download datasets
#### Usage
#### VQA questions and image dataset
Run using bash script `setup.sh`
```bash
sh setup.sh
```
Downloads and unpacks necessary datasets into `gaze_vqa_analysis/data/`

Additionally download [bottom up image features](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ)  into `data/features_val2014`

##### Eye-gaze data associated with this project:
Download [eye-gaze dataset](https://drive.google.com/file/d/1iDsjZmNVrkGG-21HqpePZMhWOrraet__/view)  and keep the csv in `data/`

### 2. Gaze to image bounding box mapping and plotting

#### Usage
```python
python3 plot_quantized_gaze.py --save_path=<save_path> --input_fixations=<input_fixations csv> --vqa_questions_path=<path to vqa questions json> --vqa_images_path=<vqa_images_path> --vqa_image_features_path=<path to image bottom up features>
```
#### Arguments and Description
|Argument |Data Type |Default |Description|
|---------|----------|--------|-----------|
|save_path|str|`./plots`|path to save generated plots|
|input_fixations|str|`data/vqa_v2_gaze_1_650_all.csv`|path to eye-gaze data csv|
|vqa_questions_path|str|`data/v2_OpenEnded_mscoco_val2014_questions.json`|path to vqa questions json|
|vqa_images_path|str|`data/val2014`|path to input images VQA V2.0|
|vqa_image_features_path|str|`data/features_val2014`|path to image bottom up features|

#### Sample Output
![sample output](https://github.com/shahansa/Gaze_vqa_analysis/blob/main/images/P1_9_448426000.png?raw=true)
### 3. Model attention visualization for MCAN
#### Usage
```python
python3 mcan_attention_visualization/img_attn_visualization.py
```

#### Useful poointers for variables used in code
* `image_attn_values.pkl` : Extracted attention from model. For a batch size of `l`, model attention is of shape `(l, 100, 1)`. 
    for more info, refer section `4.3` of [MCAN Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Deep_Modular_Co-Attention_Networks_for_Visual_Question_Answering_CVPR_2019_paper.html)
* `result_run_epoch13.pkl_31907550.json` : Run report of model, obtained by running the [MCAN Model](https://github.com/MILVLG/mcan-vqa) on [VQA-V2.0 data](https://visualqa.org/download.html). 
* `v2_OpenEnded_mscoco_val2014_questions.json` : VQA-V2.0 question data
* `question_ids.pkl` : Subset of question ids for which to plot attention, numpy array exported into pickle format. 


