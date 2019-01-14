# Convert pre-trained model from `.weights` to `.npy`

- This page explains how to load pre-trained model (DarkNet-53 and YOLOv3) from `.weights` file and store the model in a `.npy` file. 
- Since both DarkNet-53 and YOLOv3 are full convolutional networks, so only the learned parameters of convolutional layers (with learned batch normalization parameters if applicable) are stored in the pre-trained model.
- The converted pre-trained model is stored as dictionary in `.npy` file.


## Format of `.weight`
- The first four 32-bit integers are: major version, minor version, revision, images seen.
- After the first four integers, learned parameters (weights, bias and batch normalization parameters) for each convolutional layers are stored as float32 following the network layout (parsed from `.cfg` file).
- For each convolutional layer with/without batch normalization(BN), the parameters are saved with their corresponding length in the order of (check [here](https://github.com/pjreddie/darknet/blob/b13f67bfdd87434e141af532cdb5dc1b8369aa3b/src/parser.c#L958) for reference):

   | *No BN* | *With BN* |
   |:--|:--|
   |Conv Bias (out_dim)|BN beta (out_dim)|
   |Conv Weights (out_dim * in_dim * filter_height * filter_width)|BN gamma (out_dim)|
   |-|BN Moving Mean (out_dim)|
   |-|BN Moving Variance (out_dim)|
   |-|Conv Weights (out_dim * in_dim * filter_height * filter_width)|

## Format of `.npy`
- The converted model is stored as dictionary in `.npy` file. 
- First load the file as `weight_dict = np.load(NPY_PATH, encoding='latin1').item()`
- Then for each convolutional layer with name 'Layer_Name' (naming rules explained [**here**]), the parameters can be access as (if applicable):

   | *Parameters* | *Access* |
   |:--|:--|
   |Conv Weights|weight_dict[Layer_Name][weights]|
   |Conv Bias|weight_dict[Layer_Name][weights]|
   |BN gamma|weight_dict[Layer_Name][0]|
   |BN beta|weight_dict[Layer_Name][1]|
   |BN Moving Mean|weight_dict[Layer_Name][2]|
   |BN Moving Variance|weight_dict[Layer_Name][3]|
   
   
## Naming rules for convolutional layers in `.npy` file
Layer names are different from the names in `.cfg` files.
#### DarkNet-53
DarkNet-53 contains 52 convolutional layers for feature extractions named as `conv_1`...`conv_52`, and one output layer for classification named as `conv_fc_1`.
#### YOLOv3
The first 52 layers of DarkNet-53 are used as feature extractor, so the layer names are the same as DarkNet-53. YOLOv3 predicts bounding boxes at 3 different scales. There are 7 convolutional layers at each scales. Also at the second and third scale, an additional convolution layers is used for reduce the dimensionality of the input feature map. Thus, for the first scale, layers are named as `conv_1_1`...`conv_1_7`, and For the second and third scale, layers are named as `conv_2_0`...`conv_2_7` and `conv_3_0`...`conv_3_7`, where `conv_2_0` and `conv_3_0` are used for dimensionality reduction. 



## Convert models

- First download the pre-trained `.weights` file of Darknet-53 from [here](https://pjreddie.com/darknet/imagenet/) (Section 'Pre-Trained Models', Darknet53 448x448 [link](https://pjreddie.com/media/files/darknet53_448.weights)) and YOLOv3 from [here](https://pjreddie.com/darknet/yolo/) (Section 'Performance on the COCO Dataset', YOLOv3-416 [link](https://pjreddie.com/media/files/yolov3.weights)).
- Go to `experiment/convert/`

  **DarkNet-53 for image classification**

  ```
  python convert_model.py \
  	--model darknet \
  	--weights_dir WEIGHTS_DIR \
  	--save_dir SAVE_DIR
  ```
  
  
  **YOLOv3 pre-trained**
  
  ```
  python convert_model.py \
  	--model yolo \
  	--weights_dir WEIGHTS_DIR \
  	--save_dir SAVE_DIR
  ```
  
  
  **DarkNet-53 feature extractor for YOLOv3**
  
  ```
  python convert_model.py \
  	--model yolov3_feat \
  	--weights_dir WEIGHTS_DIR \
  	--save_dir SAVE_DIR
  ```
  
 
- `--weights_dir` is the directory to put the pre-trained `.weights` file and `--save_dir` is the directory to put the converted `.npy` file.
- Converted parameters are saved as `darknet53_448.npy`, `yolov3.npy` and `yolov3_feat.npy` for pre-trained DarkNet-53 classification, YOLOv3 and DarkNet-53 feature extractor.



### Reference code
- https://github.com/pjreddie/darknet
- https://github.com/qqwweee/keras-yolo3
   
## Author
Qian Ge
