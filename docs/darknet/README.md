# Darknet-53 for Image Classification
- TensorFlow implementation of nature image classification using pretrained [Darknet-53](https://pjreddie.com/darknet/imagenet/) which is used as the feature extractor in [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf).
- This experiment is used to check whether the pre-trained model is correctly converted and loaded using TensorFlow.
- The configuration of the network architecture can be found [here](https://github.com/pjreddie/darknet/blob/master/cfg/darknet53_448.cfg).
- The pretrained model (`weights` file) is first downloaded from [YOLO website](https://pjreddie.com/darknet/imagenet/) (Section Pre-Trained Models, Darknet53 448x448) and then convert to `npy` file.

## Implementation Details
- The Darknet-53 model is defined in [`src/net/darknet.py`](../../src/net/darknet.py). The network architecture is exact the same as the [configuration](https://github.com/pjreddie/darknet/blob/master/cfg/darknet53_448.cfg). For each convolutional layer with stride = 2, the input is padded by 1 pixel in both width and height before and after the input content instead of 'SAME' padding. Leaky ReLU with alpha=0.1 and batch normalization are used as for all convolutional layers except the output layer. Thus, convolutional bias only used in the output layer.
- Use [`experiment/convert/convert_model.py`](../../experiment/convert/convert_model.py) to convert the `weights` file to `npy` file. Only convolutional layer parameters are converted as Darknet-53 is a full convolutional network.
- The parameter format of convolutional layers with/without batch normalization(BN) saved in the `weights` file is as the following table (reference can be found [here](https://github.com/pjreddie/darknet/blob/b13f67bfdd87434e141af532cdb5dc1b8369aa3b/src/parser.c#L958)):

   | *No BN* | *With BN* |
   |:--|:--|
   |Conv Bias|BN beta|
   |Conv Weights|BN gamma|
   |-|BN Moving Mean|
   |-|BN Moving Variance|
   |-|Conv Weights|
- The converted model is stored as dictionary in `npy` file. Each type of parameters can be access as the following table if applicable (layers are named as `conv_1`...`conv_52`, and the output layer `conv_fc` ):

   | *Parameters* | *Access* |
   |:--|:--|
   |Conv Weights|weight_dict[Layer_Name][weights]|
   |Conv Bias|weight_dict[Layer_Name][weights]|
   |BN gamma|weight_dict[Layer_Name][0]|
   |BN beta|weight_dict[Layer_Name][1]|
   |BN Moving Mean|weight_dict[Layer_Name][2]|
   |BN Moving Variance|weight_dict[Layer_Name][3]|
- An example of image classification using the pre-trained model is in [`experiment/darknet`](../../experiment/darknet.py).


## Convert model
- Download the pre-trained `weights` file of Darknet-53 from [here](https://pjreddie.com/darknet/imagenet/) (Section Pre-Trained Models, Darknet53 448x448).
- Go to `experiment/convert/`, run

  ```
  python convert_model.py --weights_dir WEIGHTS_DIR --save_dir SAVE_DIR
  ```
 
- `--weights_dir` is the directory to put the pre-trained `weights` file and `--save_dir` is the directory to put the converted `npy` file.
- Converted parameters are saved as `darknet53_448.npy`.

## ImageNet classification
Go to `experiment/`, run

```
python darknet.py --data_dir DATA_DIR --im_name PART-OF-IMAGE-NAME --pretrained_path MODEL_PATH --rescale SHORTER_SIDE
```
- `data_dir` is the directory to put the test images. `--im_name` is the option for image names you want to test. If the testing images are all `png` files, this can be `png`. The default setting is `.jpg`.
- `--pretrained_path` is the path of pre-trained `npy` file.
- `--rescale` is the option for setting the shorter side of rescaled input image.
- The output will be the top-5 class labels and probabilities.



## Results
### Image classification using pre-trained model
- Top five predictions are shown. The probabilities are shown keeping two decimal places. Note that the pre-trained model are trained on [ImageNet](http://www.image-net.org/).
- Result of VGG19 for the same images can be found [here](https://github.com/conan7882/VGG-tensorflow#results). 
**The pre-processing of images for both experiments are the same.** 

*Data Source* | *Image* | *Result* |
|:--|:--:|:--|
[COCO](http://cocodataset.org/#home) |<img src='../../data/000000000285.jpg' height='200px'>| 1: probability: 1.00, label: brown bear, bruin, Ursus arctos<br>2: probability: 0.00, label: ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus<br>3: probability: 0.00, label: American black bear, black bear, Ursus americanus, Euarctos americanus<br>4: probability: 0.00, label: chow, chow chow<br>5: probability: 0.00, label: sloth bear, Melursus ursinus, Ursus ursinus
[COCO](http://cocodataset.org/#home) |<img src='../../data/000000000724.jpg' height='200px'>| 1: probability: 0.97, label: street sign<br>2: probability: 0.02, label: traffic light, traffic signal, stoplight<br>3: probability: 0.00, label: pole<br>4: probability: 0.00, label: parking meter<br>5: probability: 0.00, label: mailbox, letter box
[COCO](http://cocodataset.org/#home) |<img src='../../data/000000001584.jpg' height='200px'>|1: probability: 0.99, label: trolleybus, trolley coach, trackless trolley<br>2: probability: 0.01, label: school bus<br>3: probability: 0.00, label: passenger car, coach, carriage<br>4: probability: 0.00, label: fire engine, fire truck<br>5: probability: 0.00, label: minibus
[COCO](http://cocodataset.org/#home) |<img src='../../data/000000003845.jpg' height='200px'>|1: probability: 0.36, label: plate<br>2: probability: 0.23, label: burrito<br>3: probability: 0.14, label: cheeseburger<br>4: probability: 0.11, label: Dungeness crab, Cancer magister<br>5: probability: 0.05, label: potpie
[ImageNet](http://www.image-net.org/) |<img src='../../data/ILSVRC2017_test_00000004.jpg' height='200px'>|1: probability: 1.00, label: goldfish, Carassius auratus<br>2: probability: 0.00, label: tench, Tinca tinca<br>3: probability: 0.00, label: rock beauty, Holocanthus tricolor<br>4: probability: 0.00, label: anemone fish<br>5: probability: 0.00, label: puffer, pufferfish, blowfish, globefish
Self Collection | <img src='../../data/IMG_4379.jpg' height='200px'>|1: probability: 0.73, label: tabby, tabby cat<br>2: probability: 0.24, label: Egyptian cat<br>3: probability: 0.04, label: tiger cat<br>4: probability: 0.00, label: Siamese cat, Siamese<br>5: probability: 0.00, label: Persian cat
Self Collection | <img src='../../data/IMG_7940.JPG' height='200px'>|1: probability: 1.00, label: streetcar, tram, tramcar, trolley, trolley car<br>2: probability: 0.00, label: passenger car, coach, carriage<br>3: probability: 0.00, label: electric locomotive<br>4: probability: 0.00, label: trolleybus, trolley coach, trackless trolley<br>5: probability: 0.00, label: freight car

### Reference code

- https://github.com/qqwweee/keras-yolo3
   
## Author
Qian Ge
