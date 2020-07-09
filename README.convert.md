# Convertion from darknet to pytorch and tensorflow  
### See (This web pages)[https://github.com/ultralytics/yolov3#darknet-conversion]  

prepaire packages for python3,  
```
 $ python3 -m pip install torch torchvision opencv-python tqdm
```

flowing operations of web,  
```
 $ git clone https://github.com/ultralytics/yolov3 && cd yolov3
 $ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'weights/yolov3-spp.pt'
```

