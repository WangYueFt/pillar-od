# Pillar-based Object Detection for Autonomous Driving

### Prerequisite

TensorFlow (https://www.tensorflow.org/install)

TensorFlow Addons (https://www.tensorflow.org/addons/overview)

Waymo Open Dataset (https://github.com/waymo-research/waymo-open-dataset)

Lingvo (https://github.com/tensorflow/lingvo)

### Data
1. Download the data from https://waymo.com/open/

2. Pre-process the data using the script "data/generate_waymo_dataset.sh" 

### Train and eval 
Check "train.py", "eval.py", and "config.py" 

### Evaluation using pretrained models
1. Download the weights from https://drive.google.com/file/d/16cFbbKfEXc5uH7V6xDw6fy3lVBCgHLNd/view?usp=sharing

2. For car, `python eval.py --class_id=1 --nms_iou_threshold=0.7 --pillar_map_size=256 --ckpt_path=/path/to/checkpoints --data_path=/path/to/data --model_dir=/path/to/results`

   For pedestrian, `python eval.py --class_id=2 --nms_iou_threshold=0.2 --pillar_map_size=512 --ckpt_path=/path/to/checkpoints --data_path=/path/to/data --model_dir=/path/to/results`
 
If you find this repo useful for your research, please consider citing the paper

```
@inproceedings{wang2020,
      title={Pillar-based Object Detection for Autonomous Driving},
      author={Wang, Yue and Fathi, Alireza and Kundu, Abhijit and Ross, David A. and Pantofaru, Caroline and Funkhouser, Thomas A. and Solomon, Justin M.},
      booktitle={The European Conference on Computer Vision ({ECCV})},
      year={2020}
    }
```
