import cv2
import munkres
import numpy as np
import torch


# solution proposed in https://github.com/pytorch/pytorch/issues/229#issuecomment-299424875 
def flip_tensor(tensor, dim=0):
    """
    flip the tensor on the dimension dim
    """
    inv_idx = torch.arange(tensor.shape[dim] - 1, -1, -1).to(tensor.device)
    return tensor.index_select(dim, inv_idx)


#
# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
def flip_back(output_flipped, matched_parts):
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'

    output_flipped = flip_tensor(output_flipped, dim=-1)

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].clone()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints * joints_vis, joints_vis


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 1.0 * 200.0  # It was scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=interpolation
    )

    return dst_img
#
#


#
# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
def calc_dists(preds, target, normalize):
    preds = preds.type(torch.float32)
    target = target.type(torch.float32)
    dists = torch.zeros((preds.shape[1], preds.shape[0])).to(preds.device)
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                # # dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                dists[c, n] = torch.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    """
    Return percentage below threshold while ignoring values with a -1
    """
    dist_cal = torch.ne(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return torch.lt(dists[dist_cal], thr).float().sum() / num_dist_cal
    else:
        return -1


def evaluate_pck_accuracy(output, target, hm_type='gaussian', thr=0.5):
    """
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    """
    idx = list(range(output.shape[1]))
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = torch.ones((pred.shape[0], 2)) * torch.tensor([h, w],
                                                             dtype=torch.float32) / 10  # Why they divide this by 10?
        norm = norm.to(output.device)
    else:
        raise NotImplementedError
    dists = calc_dists(pred, target, norm)

    acc = torch.zeros(len(idx)).to(dists.device)
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i] = dist_acc(dists[idx[i]], thr=thr)
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    return acc, avg_acc, cnt, pred, target
#
#


#
# Operations on bounding boxes (rectangles)
def bbox_area(bbox):
    """
    Area of a bounding box (a rectangles).

    Args:
        bbox (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)

    Returns:
        float: Bounding box area.
    """
    x1, y1, x2, y2 = bbox

    dx = x2 - x1
    dy = y2 - y1

    return dx * dy


def bbox_intersection(bbox_a, bbox_b):
    """
    Intersection between two buonding boxes (two rectangles).

    Args:
        bbox_a (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)
        bbox_b (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)

    Returns:
        (:class:`np.ndarray`, float):
            Intersection limits and area.

            Format: (x_min, y_min, x_max, y_max), area
    """
    x1 = np.max((bbox_a[0], bbox_b[0]))  # Left
    x2 = np.min((bbox_a[2], bbox_b[2]))  # Right
    y1 = np.max((bbox_a[1], bbox_b[1]))  # Top
    y2 = np.min((bbox_a[3], bbox_b[3]))  # Bottom

    if x2 < x1 or y2 < y1:
        bbox_i = np.asarray([0, 0, 0, 0])
        area_i = 0
    else:
        bbox_i = np.asarray([x1, y1, x2, y2], dtype=bbox_a.dtype)
        area_i = bbox_area(bbox_i)

    return bbox_i, area_i


def bbox_union(bbox_a, bbox_b):
    """
    Union between two buonding boxes (two rectangles).

    Args:
        bbox_a (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)
        bbox_b (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)

    Returns:
        float: Union.
    """
    area_a = bbox_area(bbox_a)
    area_b = bbox_area(bbox_b)

    bbox_i, area_i = bbox_intersection(bbox_a, bbox_b)
    area_u = area_a + area_b - area_i

    return area_u


def bbox_iou(bbox_a, bbox_b):
    """
    Intersection over Union (IoU) between two buonding boxes (two rectangles).

    Args:
        bbox_a (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)
        bbox_b (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)

    Returns:
        float: Intersection over Union (IoU).
    """
    area_u = bbox_union(bbox_a, bbox_b)
    bbox_i, area_i = bbox_intersection(bbox_a, bbox_b)

    iou = area_i / area_u

    return iou
#
#


#
# Bounding box/pose similarity and association
def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[:, 0]
    yg = g[:, 1]
    vg = g[:, 2]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, :, 0]
        yd = d[n_d, :, 1]
        vd = d[n_d, :, 2]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def compute_similarity_matrices(bboxes_a, bboxes_b, poses_a, poses_b):
    assert len(bboxes_a) == len(poses_a) and len(bboxes_b) == len(poses_b)

    result_bbox = np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)
    result_pose = np.zeros((len(poses_a), len(poses_b)), dtype=np.float32)

    for i, (bbox_a, pose_a) in enumerate(zip(bboxes_a, poses_a)):
        area_bboxes_b = np.asarray([bbox_area(bbox_b) for bbox_b in bboxes_b])
        result_pose[i, :] = oks_iou(pose_a, poses_b, bbox_area(bbox_a), area_bboxes_b)
        for j, (bbox_b, pose_b) in enumerate(zip(bboxes_b, poses_b)):
            result_bbox[i, j] = bbox_iou(bbox_a, bbox_b)

    return result_bbox, result_pose


def find_person_id_associations(boxes, pts, prev_boxes, prev_pts, prev_person_ids, next_person_id=0,
                                pose_alpha=0.5, similarity_threshold=0.5, smoothing_alpha=0.):
    """
    Find associations between previous and current skeletons and apply temporal smoothing.
    It requires previous and current bounding boxes, skeletons, and previous person_ids.

    Args:
        boxes (:class:`np.ndarray`): current person bounding boxes
        pts (:class:`np.ndarray`): current human joints
        prev_boxes (:class:`np.ndarray`): previous person bounding boxes
        prev_pts (:class:`np.ndarray`): previous human joints
        prev_person_ids (:class:`np.ndarray`): previous person ids
        next_person_id (int): the id that will be assigned to the next novel detected person
            Default: 0
        pose_alpha (float): parameter to weight between bounding box similarity and pose (oks) similarity.
            pose_alpha * pose_similarity + (1 - pose_alpha) * bbox_similarity
            Default: 0.5
        similarity_threshold (float): lower similarity threshold to have a correct match between previous and
            current detections.
            Default: 0.5
        smoothing_alpha (float): linear temporal smoothing filter. Set 0 to disable, 1 to keep the previous detection.
            Default: 0.1

    Returns:
            (:class:`np.ndarray`, :class:`np.ndarray`, :class:`np.ndarray`):
                A list with (boxes, pts, person_ids) where boxes and pts are temporally smoothed.
    """
    bbox_similarity_matrix, pose_similarity_matrix = compute_similarity_matrices(boxes, prev_boxes, pts, prev_pts)
    similarity_matrix = pose_similarity_matrix * pose_alpha + bbox_similarity_matrix * (1 - pose_alpha)

    m = munkres.Munkres()
    assignments = np.asarray(m.compute((1 - similarity_matrix).tolist()))  # Munkres require a cost => 1 - similarity

    person_ids = np.ones(len(pts), dtype=np.int32) * -1
    for assignment in assignments:
        if similarity_matrix[assignment[0], assignment[1]] > similarity_threshold:
            person_ids[assignment[0]] = prev_person_ids[assignment[1]]
            if smoothing_alpha:
                boxes[assignment[0]] = (1 - smoothing_alpha) * boxes[assignment[0]] + \
                                       smoothing_alpha * prev_boxes[assignment[1]]
                pts[assignment[0]] = (1 - smoothing_alpha) * pts[assignment[0]] + \
                                     smoothing_alpha * prev_pts[assignment[1]]

    person_ids[person_ids == -1] = np.arange(next_person_id, next_person_id + np.sum(person_ids == -1))

    return boxes, pts, person_ids
#
#


#
# derived from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
def get_multi_stage_outputs(model, image,
                            with_flip=False, project2image=False, size_projected=None,
                            nof_joints=17, max_batch_size=128):
    heatmaps_avg = 0
    num_heatmaps = 0
    heatmaps = []
    tags = []

    # inference
    # outputs is a list with (default) shape
    #   [(batch, nof_joints*2, height//4, width//4), (batch, nof_joints, height//2, width//2)]
    # but it could also be (no checkpoints with this configuration)
    #   [(batch, nof_joints*2, height//4, width//4), (batch, nof_joints*2, height//2, width//2), (batch, nof_joints, height, width)]
    if len(image) <= max_batch_size:
        outputs = model(image)
    else:
        outputs = [
            torch.empty((image.shape[0], nof_joints * 2, image.shape[-2] // 4, image.shape[-1] // 4),
                        device=image.device),
            torch.empty((image.shape[0], nof_joints, image.shape[-2] // 2, image.shape[-1] // 2),
                        device=image.device)
        ]
        for i in range(0, len(image), max_batch_size):
            out = model(image[i:i + max_batch_size])
            outputs[0][i:i + max_batch_size] = out[0]
            outputs[1][i:i + max_batch_size] = out[1]

    # get higher output resolution
    higher_resolution = (outputs[-1].shape[-2], outputs[-1].shape[-1])

    for i, output in enumerate(outputs):
        if i != len(outputs) - 1:
            output = torch.nn.functional.interpolate(
                output,
                size=higher_resolution,
                mode='bilinear',
                align_corners=False
            )

        heatmaps_avg += output[:, :nof_joints]
        num_heatmaps += 1

        if output.shape[1] > nof_joints:
            tags.append(output[:, nof_joints:])

    if num_heatmaps > 0:
        heatmaps.append(heatmaps_avg / num_heatmaps)

    if with_flip:  # ToDo
        raise NotImplementedError
        # if 'coco' in cfg.DATASET.DATASET:
        #     dataset_name = 'COCO'
        # elif 'crowd_pose' in cfg.DATASET.DATASET:
        #     dataset_name = 'CROWDPOSE'
        # else:
        #     raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
        # flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER'] \
        #     if cfg.DATASET.WITH_CENTER else FLIP_CONFIG[dataset_name]
        #
        # heatmaps_avg = 0
        # num_heatmaps = 0
        # outputs_flip = model(torch.flip(image, [3]))
        # for i in range(len(outputs_flip)):
        #     output = outputs_flip[i]
        #     if len(outputs_flip) > 1 and i != len(outputs_flip) - 1:
        #         output = torch.nn.functional.interpolate(
        #             output,
        #             size=(outputs_flip[-1].size(2), outputs_flip[-1].size(3)),
        #             mode='bilinear',
        #             align_corners=False
        #         )
        #     output = torch.flip(output, [3])
        #     outputs.append(output)
        #
        #     offset_feat = cfg.DATASET.NUM_JOINTS \
        #         if cfg.LOSS.WITH_HEATMAPS_LOSS[i] else 0
        #
        #     if cfg.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
        #         heatmaps_avg += \
        #             output[:, :cfg.DATASET.NUM_JOINTS][:, flip_index, :, :]
        #         num_heatmaps += 1
        #
        #     if cfg.LOSS.WITH_AE_LOSS[i] and cfg.TEST.WITH_AE[i]:
        #         tags.append(output[:, offset_feat:])
        #         if cfg.MODEL.TAG_PER_JOINT:
        #             tags[-1] = tags[-1][:, flip_index, :, :]
        #
        # heatmaps.append(heatmaps_avg/num_heatmaps)

    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for tms in tags
        ]

    return outputs, heatmaps, tags


def aggregate_results(scale_factor, final_heatmaps, tags_list, heatmaps, tags, with_flip=False, project2image=False):
    if scale_factor == 1:
        if final_heatmaps is not None and not project2image:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    heatmaps_avg = (heatmaps[0] + heatmaps[1]) / 2.0 if with_flip else heatmaps[0]

    if final_heatmaps is None:
        final_heatmaps = heatmaps_avg
    elif project2image:
        final_heatmaps += heatmaps_avg
    else:
        final_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(final_heatmaps.size(2), final_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False
        )

    return final_heatmaps, tags_list


def transform_preds(coords, center, scale, output_size):
    # target_coords = np.zeros(coords.shape)
    target_coords = coords.copy()
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def resize(image, input_size, interpolation=cv2.INTER_LINEAR):
    h, w, _ = image.shape

    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])
    if w < h:
        w_resized = input_size
        h_resized = int((input_size / w * h + 63) // 64 * 64)
        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = input_size
        w_resized = int((input_size / h * w + 63) // 64 * 64)
        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    scale = np.array([scale_w, scale_h])
    trans = get_affine_transform(center, scale, 0, (w_resized, h_resized))

    image_resized = cv2.warpAffine(
        image,
        trans,
        (int(w_resized), int(h_resized)),
        flags=interpolation
    )

    return image_resized, center, scale


def get_multi_scale_size(image, input_size, current_scale, min_scale):
    h, w, _ = image.shape
    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])

    # calculate the size for min_scale
    min_input_size = int((min_scale * input_size + 63) // 64 * 64)
    if w < h:
        w_resized = int(min_input_size * current_scale / min_scale)
        h_resized = int(
            int((min_input_size / w * h + 63) // 64 * 64) * current_scale / min_scale
        )
        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = int(min_input_size * current_scale / min_scale)
        w_resized = int(
            int((min_input_size / h * w + 63) // 64 * 64) * current_scale / min_scale
        )
        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    return (w_resized, h_resized), center, np.array([scale_w, scale_h])


def resize_align_multi_scale(image, input_size, current_scale, min_scale, interpolation=cv2.INTER_LINEAR):
    size_resized, center, scale = get_multi_scale_size(
        image, input_size, current_scale, min_scale
    )
    trans = get_affine_transform(center, scale, 0, size_resized)

    image_resized = cv2.warpAffine(
        image,
        trans,
        size_resized,
        # (int(w_resized), int(h_resized)),
        flags=interpolation
    )

    return image_resized, size_resized, center, scale


def get_final_preds(grouped_joints, center, scale, heatmap_size):
    final_results = []
    # for each image
    for i in range(len(grouped_joints)):
        final_results.insert(i, [])
        # for each detected person
        for person in grouped_joints[i]:
            # joints = np.zeros((person.shape[0], 3))
            joints = transform_preds(person.cpu().numpy(), center, scale, heatmap_size)
            final_results[i].append(joints)

    return final_results
#
#
