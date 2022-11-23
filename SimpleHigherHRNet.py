from collections import OrderedDict

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from models.higherhrnet import HigherHRNet
from misc.HeatmapParser import HeatmapParser
from misc.utils import (get_multi_scale_size, resize_align_multi_scale, get_multi_stage_outputs, aggregate_results,
                        get_final_preds)


class SimpleHigherHRNet:
    """
    SimpleHigherHRNet class.

    The class provides a simple and customizable method to load the HigherHRNet network, load the official pre-trained
    weights, and predict the human pose on single images or a batch of images.
    """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 model_name='HigherHRNet',
                 resolution=512,
                 interpolation=cv2.INTER_LINEAR,
                 return_heatmaps=False,
                 return_bounding_boxes=False,
                 filter_redundant_poses=True,
                 max_nof_people=30,
                 max_batch_size=32,
                 device=torch.device("cpu"),
                 enable_tensorrt=False):
        """
        Initializes a new SimpleHigherHRNet object.
        HigherHRNet is initialized on the torch.device("device") and
        its pre-trained weights will be loaded from disk.

        Args:
            c (int): number of channels (when using HigherHRNet model).
            nof_joints (int): number of joints.
            checkpoint_path (str): path to an official higherhrnet checkpoint.
            model_name (str): model name (just HigherHRNet at the moment).
                Valid names for HigherHRNet are: `HigherHRNet`, `higherhrnet`
                Default: "HigherHRNet"
            resolution (int): higherhrnet input resolution - format: int == min(width, height).
                Default: 512
            interpolation (int): opencv interpolation algorithm.
                Default: cv2.INTER_LINEAR
            return_heatmaps (bool): if True, heatmaps will be returned along with poses by self.predict.
                Default: False
            return_bounding_boxes (bool): if True, bounding boxes will be returned along with poses by self.predict.
                Default: False
            filter_redundant_poses (bool): if True, redundant poses (poses being almost identical) are filtered out.
                Default: True
            max_nof_people (int): maximum number of detectable people.
                Default: 30
            max_batch_size (int): maximum batch size used in higherhrnet inference.
                Useless without multiperson=True.
                Default: 16
            device (:class:`torch.device` or str): the higherhrnet (and yolo) inference will be run on this device.
                Default: torch.device("cpu")
            enable_tensorrt (bool): Enables tensorrt inference for HigherHRnet.
                If enabled, a `.engine` file is expected as `checkpoint_path`.
                Default: False
        """

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution
        self.interpolation = interpolation
        self.return_heatmaps = return_heatmaps
        self.return_bounding_boxes = return_bounding_boxes
        self.filter_redundant_poses = filter_redundant_poses
        self.max_nof_people = max_nof_people
        self.max_batch_size = max_batch_size
        self.device = device
        self.enable_tensorrt = enable_tensorrt

        # assert nof_joints in (14, 15, 17)
        if self.nof_joints == 14:
            self.joint_set = 'crowdpose'
        elif self.nof_joints == 15:
            self.joint_set = 'mpii'
        elif self.nof_joints == 17:
            self.joint_set = 'coco'
        else:
            raise ValueError('Wrong number of joints.')

        if model_name in ('HigherHRNet', 'higherhrnet'):
            self.model = HigherHRNet(c=c, nof_joints=nof_joints)
        else:
            raise ValueError('Wrong model name.')

        if not self.enable_tensorrt:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            # fix issue with official high-resolution weights
            checkpoint = OrderedDict([(k[2:] if k[:2] == '1.' else k, v) for k, v in checkpoint.items()])
            self.model.load_state_dict(checkpoint)
            if 'cuda' in str(self.device):
                print("device: 'cuda' - ", end="")

                if 'cuda' == str(self.device):
                    # if device is set to 'cuda', all available GPUs will be used
                    print("%d GPU(s) will be used" % torch.cuda.device_count())
                    device_ids = None
                else:
                    # if device is set to 'cuda:IDS', only that/those device(s) will be used
                    print("GPU(s) '%s' will be used" % str(self.device))
                    device_ids = [int(x) for x in str(self.device)[5:].split(',')]

                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

            elif 'cpu' == str(self.device):
                print("device: 'cpu'")
            else:
                raise ValueError('Wrong device name.')

            self.model = self.model.to(device)
            self.model.eval()
        else:
            if device.type == 'cpu':
                raise ValueError('TensorRT does not support cpu device.')
            from misc.tensorrt_utils import TRTModule_HigherHRNet
            self.model = TRTModule_HigherHRNet(path=checkpoint_path, device=self.device)

        self.output_parser = HeatmapParser(num_joints=self.nof_joints,
                                           joint_set=self.joint_set,
                                           max_num_people=self.max_nof_people,
                                           ignore_too_much=True,
                                           detection_threshold=0.3)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        """
        Predicts the human pose on a single image or a stack of n images.

        Args:
            image (:class:`np.ndarray`):
                the image(s) on which the human pose will be estimated.

                image is expected to be in the opencv format.
                image can be:
                    - a single image with shape=(height, width, BGR color channel)
                    - a stack of n images with shape=(n, height, width, BGR color channel)

        Returns:
            :class:`np.ndarray` or list:
                a numpy array containing human joints for each (detected) person.

                Format:
                    if image is a single image:
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).
                    if image is a stack of n images:
                        list of n np.ndarrays with
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).

                Each joint has 3 values: (y position, x position, joint confidence).

                If self.return_heatmaps, the class returns a list with (heatmaps, human joints)
                If self.return_bounding_boxes, the class returns a list with (bounding boxes, human joints)
                If self.return_heatmaps and self.return_bounding_boxes, the class returns a list with
                    (heatmaps, bounding boxes, human joints)
        """
        if len(image.shape) == 3:
            return self._predict_single(image)
        elif len(image.shape) == 4:
            return self._predict_batch(image)
        else:
            raise ValueError('Wrong image format.')

    def _predict_single(self, image):
        ret = self._predict_batch(image[None, ...])
        if len(ret) > 1:  # heatmaps and/or bboxes and joints
            ret = [r[0] for r in ret]
        else:  # joints only
            ret = ret[0]
        return ret

    def _predict_batch(self, image):
        with torch.no_grad():

            heatmaps_list = None
            tags_list = []

            # scales and base (size, center, scale)
            scales = (1,)  # ToDo add support to multiple scales

            scales = sorted(scales, reverse=True)
            base_size, base_center, base_scale = get_multi_scale_size(
                image[0], self.resolution, 1, 1
            )

            # for each scale (at the moment, just one scale)
            for idx, scale in enumerate(scales):
                # rescale image, convert to tensor, move to device
                images = list()
                for img in image:
                    image, size_resized, _, _ = resize_align_multi_scale(
                        img, self.resolution, scale, min(scales), interpolation=self.interpolation
                    )
                    image = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
                    image = image.to(self.device)
                    images.append(image)
                images = torch.cat(images)

                # inference
                # output: list of HigherHRNet outputs (heatmaps)
                # avg_heatmaps: averaged heatmaps
                # tags: per-pixel identity ids.
                #       See Newell et al., Associative Embedding: End-to-End Learning for Joint Detection and
                #           Grouping, NIPS 2017. https://arxiv.org/abs/1611.05424 or
                #           http://papers.nips.cc/paper/6822-associative-embedding-end-to-end-learning-for-joint-detection-and-grouping
                outputs, heatmaps, tags = get_multi_stage_outputs(
                    self.model, images, with_flip=False, project2image=True, size_projected=size_resized,
                    nof_joints=self.nof_joints, max_batch_size=self.max_batch_size
                )

                # aggregate the multiple heatmaps and tags
                heatmaps_list, tags_list = aggregate_results(
                    scale, heatmaps_list, tags_list, heatmaps, tags, with_flip=False, project2image=True
                )

            heatmaps = heatmaps_list.float() / len(scales)
            tags = torch.cat(tags_list, dim=4)

            # refine prediction
            # grouped has the shape (people, joints, 4) -> 4: (x, y, confidence, tag)
            # scores has the shape (people, ) and corresponds to the person confidence before refinement
            grouped, scores = self.output_parser.parse(
                heatmaps, tags, adjust=True, refine=True  # ToDo parametrize these two parameters
            )

            # get final predictions
            final_results = get_final_preds(
                grouped, base_center, base_scale, [heatmaps.shape[3], heatmaps.shape[2]]
            )

            if self.filter_redundant_poses:
                # filter redundant poses - this step filters out poses whose joints have, on average, a difference
                #   lower than 3 pixels
                # this is useful when refine=True in self.output_parser.parse because that step joins together
                #   skeleton parts belonging to the same people (but then it does not remove redundant skeletons)
                final_pts = []
                # for each image
                for i in range(len(final_results)):
                    final_pts.insert(i, list())
                    # for each person
                    for pts in final_results[i]:
                        if len(final_pts[i]) > 0:
                            diff = np.mean(np.abs(np.array(final_pts[i])[..., :2] - pts[..., :2]), axis=(1, 2))
                            if np.any(diff < 3):  # average diff between this pose and another one is less than 3 pixels
                                continue
                        final_pts[i].append(pts)
                final_results = final_pts

            pts = []
            boxes = []
            for i in range(len(final_results)):
                pts.insert(i, np.asarray(final_results[i]))
                if len(pts[i]) > 0:
                    pts[i][..., [0, 1]] = pts[i][..., [1, 0]]  # restoring (y, x) order as in SimpleHRNet
                    pts[i] = pts[i][..., :3]

                    if self.return_bounding_boxes:
                        left_top = np.min(pts[i][..., 0:2], axis=1)
                        right_bottom = np.max(pts[i][..., 0:2], axis=1)
                        # [x1, y1, x2, y2]
                        boxes.insert(i, np.stack(
                            [left_top[:, 1], left_top[:, 0], right_bottom[:, 1], right_bottom[:, 0]], axis=-1
                        ))
                else:
                    boxes.insert(i, [])

        res = list()
        if self.return_heatmaps:
            res.append(heatmaps)
        if self.return_bounding_boxes:
            res.append(boxes)
        res.append(pts)

        if len(res) > 1:
            return res
        else:
            return res[0]


if __name__ == '__main__':
    hhrnet = SimpleHigherHRNet(
        c=32, nof_joints=17, checkpoint_path='./weights/pose_higher_hrnet_w32_512.pth',
        resolution=512, device='cuda'
    )
    # img = np.ones((384, 256, 3), dtype=np.uint8)

    import cv2
    img = cv2.imread('./sample.jpg', cv2.IMREAD_ANYCOLOR)

    hhrnet.predict(img)
