# CNN Car Object Detection

## Introduction
Trained a Convolutional Neural Network using Pytorch, YoloV5, and Transfer Learning Techniques to be able to detect cars within an image. The model was trained using 15,000 labeled images of cars during various times of the day. The model achieved a MAP of 0.6727.

## My Approach
For my approach, I chose to use YoloV5 for my transfer learning. A key reason why I chose this model is that it has a repository already implemented which allows for the fast creation of a model. YoloV5 has multiple versions of various sizes: Yolov5n, Yolov5s, Yolov5m, Yolov5I, and Yolov5x. For this project, I chose Yolov5x6 because it has the largest size of 140.7 million parameters which will allow for the model to detect more complex features and pair well with the large dataset given. Although YoloV5 does not have as accurate detection as Faster RNN, it has a faster inference time which allows for more testing. 

## Parameter Choice
For the hyper-parameters, I used the parameters given in the pretrain model: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, and copy_paste=0.0. 



## Results

#### Figure 1
<img src="https://github.com/jayteaftw/CNN-Car-Object-Detection/blob/main/graphs/image1.png" height="400" />

#### Figure 2
<img src="https://github.com/jayteaftw/CNN-Car-Object-Detection/blob/main/graphs/image2.png" height="400" />

#### Figure 3
<img src="https://github.com/jayteaftw/CNN-Car-Object-Detection/blob/main/graphs/image3.png" height="400" />

#### Figure 4
<img src="https://github.com/jayteaftw/CNN-Car-Object-Detection/blob/main/graphs/image4.png" height="400" />

#### Figure 5
<img src="https://github.com/jayteaftw/CNN-Car-Object-Detection/blob/main/graphs/image5.png" height="400" />

#### Figure 6
<img src="https://github.com/jayteaftw/CNN-Car-Object-Detection/blob/main/graphs/image6.png" height="400" />

#### Figure 7
<img src="https://github.com/jayteaftw/CNN-Car-Object-Detection/blob/main/graphs/image7.png" height="400" />

#### Figure 8
<img src="https://github.com/jayteaftw/CNN-Car-Object-Detection/blob/main/graphs/image8.png" height="400" />

#### Figure 9
<img src="https://github.com/jayteaftw/CNN-Car-Object-Detection/blob/main/graphs/image9.png" height="400" />


## Analysis
I tested multiple layers frozen. The best combination was 15 layers frozen, batch size of 16 and 100 epochs as seen in Figures 1 through 3 which had a final mean average precision of 0.6727. However, the training loss line for cls seen in figure 3 decreases significantly compared to the validation loss implying that there is some overfitting. However, the training loss converged for the box with the validation loss which implies some underfitting occurred. These events happened around the 60 epoch mark. To account for this, I used the best.pt weights which select the best weights that did not overfit. For the other two submissions, I chose to freeze 10 layers with 100 epochs and a batch size of 16 as seen in Figures 4 through 6, and to freeze 10 layers with a batch size of 64 as seen in Figures 7 through 9. The 10 frozen performed worse than 15 which seems to show that the transfer learning model performs better when some of the lower level features are kept untouched. Surprisingly, the batch size of 64 
