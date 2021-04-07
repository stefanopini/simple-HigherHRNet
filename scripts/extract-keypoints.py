import os
import sys
import argparse
import csv
import cv2
import json
import time
import torch

sys.path.insert(1, os.getcwd())
from SimpleHigherHRNet import SimpleHigherHRNet
from misc.visualization import check_video_rotation


def main(format, filename, hrnet_c, hrnet_j, hrnet_weights, image_resolution, max_nof_people, max_batch_size,
         csv_output_filename, csv_delimiter, json_output_filename, device):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # print(device)

    rotation_code = check_video_rotation(filename)
    video = cv2.VideoCapture(filename)
    assert video.isOpened()
    nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    assert format in ('csv', 'json')
    if format == 'csv':
        assert csv_output_filename.endswith('.csv')
        fd = open(csv_output_filename, 'wt', newline='')
        csv_output = csv.writer(fd, delimiter=csv_delimiter)
    elif format == 'json':
        assert json_output_filename.endswith('.json')
        fd = open(json_output_filename, 'wt')
        json_data = {}

    model = SimpleHigherHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        resolution=image_resolution,
        max_nof_people=max_nof_people,
        max_batch_size=max_batch_size,
        device=device
    )

    index = 0
    while True:
        t = time.time()

        ret, frame = video.read()
        if not ret:
            break
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        pts = model.predict(frame)

        # csv format is:
        #   frame_index,detection_index,<point 0>,<point 1>,...,<point hrnet_j>
        # where each <point N> corresponds to three elements:
        #   y_coordinate,x_coordinate,confidence

        # json format is:
        #   {frame_index: [[<point 0>,<point 1>,...,<point hrnet_j>], ...], ...}
        # where each <point N> corresponds to three elements:
        #   [y_coordinate,x_coordinate,confidence]

        if format == 'csv':
            for j, pt in enumerate(pts):
                row = [index, j] + pt.flatten().tolist()
                csv_output.writerow(row)
        elif format == 'json':
            json_data[index] = list()
            for j, pt in enumerate(pts):
                json_data[index].append(pt.tolist())

        fps = 1. / (time.time() - t)
        print('\rframe: % 4d / %d - framerate: %f fps ' % (index, nof_frames - 1, fps), end='')

        index += 1

    if format == 'json':
        json.dump(json_data, fd)

    fd.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract and save keypoints in csv or json format.\n'
                    'csv format is:\n'
                    '  frame_index,detection_index,<point 0>,<point 1>,...,<point hrnet_j>\n'
                    'where each <point N> corresponds to three elements:\n'
                    '  y_coordinate,x_coordinate,confidence\n'
                    'json format is:\n'
                    '  {frame_index: [[<point 0>,<point 1>,...,<point hrnet_j>], ...], ...}\n'
                    'where each <point N> corresponds to three elements:\n'
                    '[y_coordinate,x_coordinate,confidence]')
    parser.add_argument("--format", help="output file format. CSV or JSON.",
                        type=str, default=None)
    parser.add_argument("--filename", "-f", help="open the specified video",
                        type=str, default=None)
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=32)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_higher_hrnet_w32_512.pth")
    parser.add_argument("--image_resolution", "-r", help="image resolution (`512` or `640`)", type=int, default=512)
    parser.add_argument("--max_nof_people", help="maximum number of visible people", type=int, default=30)
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--csv_output_filename", help="filename of the csv that will be written.",
                        type=str, default='output.csv')
    parser.add_argument("--csv_delimiter", help="csv delimiter", type=str, default=',')
    parser.add_argument("--json_output_filename", help="filename of the json file that will be written.",
                        type=str, default='output.json')
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
