# Multi-person Human Pose Estimation with HigherHRNet in PyTorch

This is an unofficial implementation of the paper
[*HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation*](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_HigherHRNet_Scale-Aware_Representation_Learning_for_Bottom-Up_Human_Pose_Estimation_CVPR_2020_paper.pdf).  
The code is a simplified version of the [official code](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
 with the ease-of-use in mind.

The code is fully compatible with the
 [official pre-trained weights](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation).
 It supports both Windows and Linux.

This repository currently provides:
- A slightly simpler implementation of ``HigherHRNet`` in PyTorch (>=1.0) - compatible with official weights 
(``pose_higher_hrnet_*``).
- A simple class (``SimpleHigherHRNet``) that loads the HigherHRNet network for the bottom-up human pose 
estimation, loads the pre-trained weights, and make human predictions on a single image or a batch of images.
- Support for multi-GPU inference.
- Multi-person support by design (HigherHRNet is a bottom-up approach).
- A reference code that runs a live demo reading frames from a webcam or a video file.

This repository is built along the lines of the repository
[*simple-HRNet*](https://github.com/stefanopini/simple-HRNet).  
Unfortunately, compared to HRNet, results and performance of HigherHRNet are somewhat disappointing: the network and 
the required post-processing are slower and the predictions does not look more precise. 
Moreover, multiple skeletons are often predicted for the same person, requiring additional steps to filter out the
redundant poses.  
On the other hand, being a bottom-up approach, HigherHRNet does not rely on any person detection algorithm like Yolo-v3
and can be used for person detection too.
 
### Examples

<table>
 <tr>
  <td align="center"><img src="./gifs/gif-01-output.gif" width="100%" height="auto" /></td>
  <td align="center"><img src="./gifs/gif-02-output.gif" width="100%" height="auto" /></td>
 </tr>
</table>

### Class usage

```
import cv2
from SimpleHigherHRNet import SimpleHigherHRNet

model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
image = cv2.imread("image.png", cv2.IMREAD_COLOR)

joints = model.predict(image)
```

The most useful parameters of the `__init__` function are:
<table>
 <tr>
  <td>c</td><td>number of channels (HRNet: 32, 48)</td>
 </tr>
 <tr>
  <td>nof_joints</td><td>number of joints (COCO: 17, CrowdPose: 14)</td>
 </tr>
 <tr>
  <td>checkpoint_path</td><td>path of the (official) weights to be loaded</td>
 </tr>
 <tr>
  <td>resolution</td><td>image resolution (min side), it depends on the loaded weights</td>
 </tr>
 <tr>
  <td>return_heatmaps</td><td>the `predict` method returns also the heatmaps</td>
 </tr>
 <tr>
  <td>return_bounding_boxes</td><td>the `predict` method returns also the bounding boxes</td>
 </tr>
 <tr>
  <td>filter_redundant_poses</td><td>redundant poses (poses being almost identical) are filtered out</td>
 </tr>
 <tr>
  <td>max_nof_people</td><td>maximum number of people in the scene</td>
 </tr>
 <tr>
  <td>max_batch_size</td><td>maximum batch size used in hrnet inference</td>
 </tr>
 <tr>
  <td>device</td><td>device (cpu or cuda)</td>
 </tr>
</table>

### Running the live demo

From a connected camera:
```
python scripts/live-demo.py --camera_id 0
```
From a saved video:
```
python scripts/live-demo.py --filename video.mp4
```

For help:
```
python scripts/live-demo.py --help
```

### Installation instructions

- Clone the repository  
 ``git clone https://github.com/stefanopini/simple-HigherHRNet.git``
- Install the required packages  
 ``pip install -r requirements.txt``
- Download the official pre-trained weights from 
[https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
  
  Direct links, COCO ([official Drive folder](https://drive.google.com/drive/folders/1X9-TzWpwbX2zQf2To8lB-ZQHMYviYYh6)):
  - w48 640 (more accurate, but slower)   
    [pose_higher_hrnet_w48_640.pth.tar](https://drive.google.com/file/d/10j9Wx_I2H6qaw-prAdlJ44fLryDtA-ah/view)
  - w32 640 (less accurate, but faster)  
    [pose_higher_hrnet_w32_640.pth.tar](https://drive.google.com/file/d/1uEcQlm1rjV-JRgVbaP79Y5sMLqUX2ciD/view)
  - w32 512 (even less accurate, but even faster) - Used as default in `live_demo.py`  
    [pose_higher_hrnet_w32_512.pth](https://drive.google.com/file/d/1V9Iz0ZYy9m8VeaspfKECDW0NKlGsYmO1/view)
  
  Remember to set the parameters of SimpleHigherHRNet accordingly (in particular `c` and `resolution`).
- Your folders should look like:
    ```
    simple-HigherHRNet
    ├── gifs                    (preview in README.md)
    ├── misc                    (misc)
    ├── models                  (pytorch models)
    ├── scripts                 (scripts)
    └── weights                 (HigherHRnet weights)
    ```

### ToDos
- [ ] Add keypoint extraction script
- [ ] Optimize the post-processing steps
- [ ] Add COCO dataset and evaluation
- [ ] Add Train/Test scripts
