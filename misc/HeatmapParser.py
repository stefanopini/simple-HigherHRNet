import munkres
import numpy as np
import torch
from collections import defaultdict

from misc import visualization


def py_max_match(scores):
    m = munkres.Munkres()
    assoc = m.compute(scores)
    assoc = np.array(assoc).astype(np.int32)
    return assoc


# derived from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/
class HeatmapParser(object):
    def __init__(self,
                 num_joints=17,
                 joint_set='coco',
                 max_num_people=30,
                 nms_kernel=5, nms_stride=1, nms_padding=2,
                 detection_threshold=0.1, tag_threshold=1., use_detection_val=True, ignore_too_much=True
                 ):
        """
        Heatmap Parser running on pytorch
        """
        assert joint_set in ('coco', 'crowdpose')

        self.num_joints = num_joints
        self.joint_set = joint_set
        self.max_num_people = max_num_people
        self.tag_per_joint = True
        self.maxpool = torch.nn.MaxPool2d(nms_kernel, nms_stride, nms_padding)
        self.detection_threshold = detection_threshold
        self.tag_threshold = tag_threshold
        self.use_detection_val = use_detection_val
        self.ignore_too_much = ignore_too_much

    def nms(self, det):
        maxm = self.maxpool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def match_by_tag_torch(self, data):
        joint_order = visualization.joints_dict()[self.joint_set]['order']

        tag_k, loc_k, val_k = data
        device = tag_k.device
        default_ = torch.zeros((self.num_joints, 3 + tag_k.shape[2]), device=device)

        loc_k = loc_k.float()
        joint_k = torch.cat((loc_k, val_k[..., None], tag_k), dim=2)  # nx30x2, nx30x1, nx30x1

        joint_dict = defaultdict(lambda: default_.clone().detach())
        tag_dict = {}
        for i in range(self.num_joints):
            idx = joint_order[i]

            tags = tag_k[idx]
            joints = joint_k[idx]
            mask = joints[:, 2] > self.detection_threshold
            tags = tags[mask]
            joints = joints[mask]

            if joints.shape[0] == 0:
                continue

            if i == 0 or len(joint_dict) == 0:
                for tag, joint in zip(tags, joints):
                    key = tag[0]
                    joint_dict[key.item()][idx] = joint
                    tag_dict[key.item()] = [tag]
            else:
                grouped_keys = list(joint_dict.keys())[:self.max_num_people]
                grouped_tags = [torch.mean(torch.as_tensor(tag_dict[i]), dim=0, keepdim=True) for i in grouped_keys]

                if self.ignore_too_much and len(grouped_keys) == self.max_num_people:
                    continue

                grouped_tags = torch.as_tensor(grouped_tags, device=device)
                if len(grouped_tags.shape) < 2:
                    grouped_tags = grouped_tags.unsqueeze(0)

                diff = joints[:, None, 3:] - grouped_tags[None, :, :]
                diff_normed = torch.norm(diff, p=2, dim=2)
                diff_saved = diff_normed.clone().detach()

                if self.use_detection_val:
                    diff_normed = torch.round(diff_normed) * 100 - joints[:, 2:3]

                num_added = diff.shape[0]
                num_grouped = diff.shape[1]

                if num_added > num_grouped:
                    diff_normed = torch.cat(
                        (diff_normed, torch.zeros((num_added, num_added - num_grouped), device=device) + 1e10),
                        dim=1
                    )

                pairs = py_max_match(diff_normed.detach().cpu().numpy())
                for row, col in pairs:
                    if (
                            row < num_added
                            and col < num_grouped
                            and diff_saved[row][col] < self.tag_threshold
                    ):
                        key = grouped_keys[col]
                        joint_dict[key][idx] = joints[row]
                        tag_dict[key].append(tags[row])
                    else:
                        key = tags[row][0].item()
                        joint_dict[key][idx] = joints[row]
                        tag_dict[key] = [tags[row]]

        # # added to correctly limit the overall number of people
        # # this shouldn't be needed if self.ignore_too_much is True
        # if len(joint_dict.keys()) > self.max_num_people:
        #     # create a dictionary with {confidence: joint_dict key}
        #     joint_confidence = {torch.mean(v[:, 2]).item(): k for k, v in joint_dict.items()}
        #     # filter joint_dict to keep the first self.max_num_people elements with higher joint confidence
        #     joint_dict = {joint_confidence[k]: joint_dict[joint_confidence[k]]
        #                   for k in sorted(joint_confidence.keys(), reverse=True)[:self.max_num_people]}

        # ret = torch.tensor([joint_dict[i] for i in joint_dict], dtype=torch.float32, device=device)
        if len(joint_dict) > 0:
            ret = torch.stack([joint_dict[i] for i in joint_dict])
        else:
            # if no people are detected, return a tensor with size 0
            size = list(default_.size())
            size.insert(0, 0)
            ret = torch.zeros(size)
        return ret

    def match_torch(self, tag_k, loc_k, val_k):
        match = lambda x: self.match_by_tag_torch(x)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def top_k_torch(self, det, tag):
        # det = torch.Tensor(det, requires_grad=False)
        # tag = torch.Tensor(tag, requires_grad=False)

        det = self.nms(det)
        num_images = det.size(0)
        num_joints = det.size(1)
        h = det.size(2)
        w = det.size(3)
        det = det.view(num_images, num_joints, -1)
        val_k, ind = det.topk(self.max_num_people, dim=2)

        tag = tag.view(tag.size(0), tag.size(1), w * h, -1)
        if not self.tag_per_joint:
            tag = tag.expand(-1, self.num_joints, -1, -1)

        tag_k = torch.stack(
            [
                torch.gather(tag[:, :, :, i], 2, ind)
                for i in range(tag.size(3))
            ],
            dim=3
        )

        # added to reduce the number of unique tags
        tag_k = (tag_k * 10).round() / 10  # ToDo parametrize this

        x = ind % w
        y = (ind // w).long()

        ind_k = torch.stack((x, y), dim=3)

        ret = {
            'tag_k': tag_k,
            'loc_k': ind_k,
            'val_k': val_k
        }

        return ret

    def adjust_torch(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        # print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx + 1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id, 0] = y + 0.5
                        ans[batch_id][people_id, joint_id, 1] = x + 0.5
        return ans

    def refine_torch(self, det, tag, keypoints):
        """
        Given initial keypoint predictions, we identify missing joints
        :param det: torch.tensor of size (17, 128, 128)
        :param tag: torch.tensor of size (17, 128, 128) if not flip
        :param keypoints: torch.tensor of size (17, 4) if not flip, last dim is (x, y, det score, tag score)
        :return:
        """
        if len(tag.shape) == 3:
            # tag shape: (17, 128, 128, 1)
            tag = tag[:, :, :, None]

        tags = []
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].type(torch.int32)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = torch.tensor(tags, device=tag.device).mean(dim=0, keepdim=True)
        ans = []

        for i in range(keypoints.shape[0]):
            # score of joints i at all position
            tmp = det[i, :, :]
            # distance of all tag values with mean tag of current detected people
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(dim=2) ** 0.5)
            tmp2 = tmp - torch.round(tt)

            def unravel_index(index, shape):
                out = []
                for dim in reversed(shape):
                    out.append(index % dim)
                    index = index // dim
                return tuple(reversed(out))

            # find maximum position
            y, x = unravel_index(torch.argmax(tmp2), tmp.shape)
            xx = x.clone().detach()
            yy = y.clone().detach()
            x = x.float()
            y = y.float()
            # detection score at maximum position
            val = tmp[yy, xx]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = torch.tensor(ans)

        if ans is not None:
            for i in range(det.shape[0]):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                    # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i, :2]
                    keypoints[i, 2] = ans[i, 2]

        return keypoints

    def parse(self, det, tag, adjust=True, refine=True):
        ans = self.match_torch(**self.top_k_torch(det, tag))

        if adjust:
            ans = self.adjust_torch(ans, det)

        scores = [i[:, 2].mean() for i in ans[0]]

        if refine:
            # for each image
            for i in range(len(ans)):
                # for each detected person
                for j in range(len(ans[i])):
                    det_ = det[i]
                    tag_ = tag[i]
                    ans_ = ans[i][j]
                    if not self.tag_per_joint:
                        tag_ = torch.repeat(tag_, (self.num_joints, 1, 1, 1))
                    ans[i][j] = self.refine_torch(det_, tag_, ans_)
            # after this refinement step, there may be multiple detections with almost identical keypoints...
            # an attempt to aggregate them is done afterwards in SimpleHigherHRNet

        return ans, scores
