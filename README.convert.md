# Convertion from darknet to pytorch and tensorflow  
### Folowing [This web pages](https://github.com/ultralytics/yolov3#darknet-conversion)  

install torch following [pytorch](https://pytorch.org/get-started/locally/),  
```
 $ python3 -m pip install torch==1.5.1+cu92 torchvision==0.6.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
 $ python3 -c "import torch;print(torch.cuda.is_available())"
 True
```

prepaire miscellaneous packages for python3 according to own conditons,  
```
 $ python3 -m pip install opencv-python tqdm
```

flowing operations of web,  
```
 $ git clone https://github.com/ultralytics/yolov3 && cd yolov3
 $ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'weights/yolov3-spp.pt'
```

Testing for coco/5k.txt.  
prepaire coco dataset for test, write 5k.txt and,  
```
 $ python3 test.py --cfg yolov3-spp.cfg --weights yolov3-spp-ultralytics.pt --img 640 --augment
```
