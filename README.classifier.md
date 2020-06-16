# Backbone classifier from Dataset download unil estimation mAP  

## Backbone Network : EfficientNetB0  

cfg/efficientnet_b0.cfg may be implemented by [AlexeyAB darknet](https://github.com/AlexeyAB/darknet).  
And trained with ImageNet Dataset.  

Pre-trained weights is provided on [Cross Stage Partial Networks](https://github.com/WongKinYiu/CrossStagePartialNetworks).  

## To improve enet4y2-coco.cfg performance
To improve mAP or to customize target object categories backbone network should be trained from scratch.  
I use [ImageNet_Utils](https://github.com/k5iogura/ImageNet_Utils) to collect my target categories from ImageNet urls.  
I collect ImageNet urls related to VOC Keywords and download images.  

- clone repo,  
```
$ git clone --recursive https://github.com/tzutalin/ImageNet_Utils.git
```

- generate url list and download it.  
```
$ ./inet4voc.sh
$ ./downloadutils.py --downloadImages --wnid_list imagenet.labels.safedomain900.list -n 1000 -th 50
$ ./make_trainval.sh
```

1000 urls/category are collected.  
50 CPU threads are used to download.  
Images will be saved under 'inet_image/*category_id*/' directory.  
*category_id* is like n01592540, n03505504 so on.  

- train classifier  

write image.data like below,  
```
classes=248
train  = ImageNet_Utils/train.txt
valid  = ImageNet_Utils/valid.txt
backup = backupEffb0
labels = ImageNet_Utils/inet-voc.labels.list
names  = ImageNet_Utils/imagenet.shortnames.safedomain900.list
top=5
```
248 classes is example and denotes categories related to VOC 20 categories in ImageNet url list.  

start training  
```
$ ./darknet classifier train image.data cfg/efficientnet_b0.cfg
```
Unfortunately wait over 20 days even if on double GTX1080Ti.  

