# face-mask-det

Download the following weights file to run the script

weights file : https://drive.google.com/file/d/1-A0a5tgg8pvW0nOuMB8wk2wXFtWc_64V/view?usp=sharing



## Model Details: 

 detections_count = 5262, unique_truth_count = 3029  
 
class_id = 0, name = mask, ap = 96.20%   	 (TP = 2385, FP = 152) 

class_id = 1, name = not_covered, ap = 96.33%   	 (TP = 185, FP = 11) 

class_id = 2, name = no_mask, ap = 94.22%   	 (TP = 293, FP = 32) 


 for conf_thresh = 0.25, precision = 0.94, recall = 0.95, F1-score = 0.94 
 
 for conf_thresh = 0.25, TP = 2863, FP = 195, FN = 166, average IoU = 76.25 % 
 

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
 
 mean average precision (mAP@0.50) = 0.955798, or 95.58 % 
 
Total Detection Time: 20 Seconds


## Dataset:

853 images were used in training the model and is avaiable in kaggle.com

Dataset with annotation : https://drive.google.com/file/d/1U1UV74YXwIcl4dirA8LnxOT43HljxHxs/view?usp=sharing





