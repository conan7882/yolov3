# YOLOv3 for Object Detection

- TensorFlow implementation of [YOLOv3](https://arxiv.org/abs/1804.02767) for object detection.
- Both inference and training pipelines implemented.
- For inference using pre-trained model, the model stored in `.weights` file is first downloaded from official [YOLO website](https://pjreddie.com/darknet/yolo/) (Section 'Performance on the COCO Dataset', YOLOv3-416 [link](https://pjreddie.com/media/files/yolov3.weights)), then [converted](docs/convert.md) to `.npy` file and finally loaded by the TensorFlow model for prediction.
- For training, the pre-trained DarkNet-53 is used as the feature extractor and the YOLO prediction layers at three scales are trained from scratch. Data augmentation such as random flipping, cropping, resize, affine transformation and color change (hue, saturation, brightness) are applied. Anchor clustering and multiple scale training (rescale training images every 10 epochs) are implemented as well.



## TODO

- [x] Convert pre-trained `.weights` model to `.npy` file [detail](docs/convert.md)
- [x] Pre-trained DarkNet-53 for image classification [detail](docs/darknet.md)
- [x] Object detection using pre-trained YOLOv3 trained on [COCO](http://cocodataset.org/#home) dataset **detail**
- [x] YOLOv3 training pipeline
- [x] Train on VOC dataset
- [ ] Evaluation
- [ ] Train on custom dataset 

## Requirements

## Implementation details

## Use pre-trained model for object detection
### Convert pre-trained model
- Download the pre-trained model from [here](https://pjreddie.com/darknet/yolo/) (Section 'Performance on the COCO Dataset', YOLOv3-416 [link](https://pjreddie.com/media/files/yolov3.weights))
- To convert the model stored in `.weights` file to `.npy` file, go to `experiment/convert/`, run
 
  ```
  python convert_model.py \
  	--model yolo \
  	--weights_dir WEIGHTS_DIR \
  	--save_dir SAVE_DIR
  ```
- `--weights_dir` is the directory to put the pre-trained `.weights` file and `--save_dir` is the directory to put the converted `.npy` file.
- Converted parameters are saved as `yolov3.npy`.
- More details for converting model can be found [here](docs/convert.md).

### Setup configuration
- Modified the config file `configs/pretrain_coco_path.cfg` with the following content: 

  	```
  	[path]
	coco_pretrained_npy = DIRECTORY/TO/MODEL/yolov3.npy
	save_path = DIRECTORY/TO/SAVE/RESULT/
	test_image_path = DIRECTORY/OF/TEST/IMAGE/
	test_image_name = .jpg
  	```
  	
	- Put converted pretrained model `yolov3.npy` in `coco_pretrained_npy`.
	- Put testing images in `test_image_path`.
	- Part of testimg image names is specified by `test_image_name`.
	- Result images will be saved in `save_path`.
	
- Use `obj_score_thresh` and `nms_iou_thresh` in config file `configs/coco80.cfg` to setup the parameters of non-maximum suppression to remove multiple bounding boxes for one detected object. 
  - `obj_score_thresh` is the threshold for deciding if a bounding box detects an object class based on the score.
  - `nms_iou_thresh` is the threshold for deciding if two bounding boxes overlap too much based on the IoU.
  
	

### Prediction
- Put testing images in `test_image_path` in `pretrain_coco_path.cfg` and go to `experiment\`, run

  ```
  python yolov3.py --detect
  ```
- Result images will be saved in `save_path` setting in `configs/pretrain_coco_path.cfg`.

### Sample results
<img src='docs/figs/pretrained/im0.png' height='350px'>
<img src='docs/figs/pretrained/im1.png' height='350px'>
<img src='docs/figs/pretrained/im2.png' height='350px'>
<img src='docs/figs/pretrained/im3.png' height='400px'>
<img src='docs/figs/pretrained/im4.png' height='400px'>

### Reference code

- https://github.com/pjreddie/darknet
- https://github.com/experiencor/keras-yolo3
- https://github.com/qqwweee/keras-yolo3
   
## Author
Qian Ge
