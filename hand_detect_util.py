# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path
import os

import numpy as np
import torch

import cv2
import open3d as o3d

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.plots import colors, plot_one_box
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def generate_point_cloud_from_depth(depht_val,img_width,img_height,depth_threshold=550,is_visualize=False, voxel_size=5.0, dbscan_eps=8.0,use_voxel_downsample=True):
    # ===== crop all parts too far from camera =====
    depht_val[depht_val>depth_threshold] = 0

    if is_visualize:
        vis_depth = depht_val / 256.0
        cv2.imshow("Depth image", vis_depth)
        cv2.waitKey(0)

    # ===== convert numpy array to open3d image =====
    image = o3d.geometry.Image(depht_val.astype(np.uint16))
    intrinsic_mat = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, 475.065948, 475.065857, 315.944855, 245.287079)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(image, intrinsic_mat, depth_scale=1.0)

    if use_voxel_downsample:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if is_visualize:
        o3d.visualization.draw_geometries([pcd])

    pcd_points = np.asarray(pcd.points)
    if len(pcd_points) == 0:
        print("Cannot generate point cloud")
        return None

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=10, print_progress=True))

    labels = DBSCAN(eps=dbscan_eps,min_samples=10,algorithm='kd_tree').fit(pcd_points).labels_

    if len(labels) == 0:
        print("Cannot find any label")
        return None

    if is_visualize:
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd])

    counter = Counter(labels)
    cluster_distance = {}

    for point_idx, point in enumerate(pcd.points):
        label = labels[point_idx]
        if counter[label] < 100: continue
        
        if not label in cluster_distance:
            cluster_distance[label] = 0
        
        distance = np.sum(point**2)
        cluster_distance[label] += distance

    for label in cluster_distance:
        cluster_distance[label] /= counter[label]

    if len(cluster_distance) != 0:
        hand_label = min(cluster_distance, key=cluster_distance.get)

        pcd.points = o3d.utility.Vector3dVector([point for (i, point) in enumerate(pcd_points) if labels[i] == hand_label])
        pcd_points = np.asarray(pcd.points)

        if is_visualize:
            o3d.visualization.draw_geometries([pcd])

    # farthest point sampling
    def cal_dis(p0, points):
        return ((p0 - points)**2).sum(axis=1)

    def farthest_point_sampling(pts, k):
        if len(pts) < k:
            return [i for i in range(len(pts))] + [np.random.randint(len(pts)) for _ in range(k - len(pts))]

        indices = np.zeros((k, ), dtype=np.uint32)
        indices[0] = np.random.randint(len(pts))
        min_distances = cal_dis(pts[indices[0]], pts)
        for i in range(1, k):
            indices[i] = np.argmax(min_distances)
            min_distances = np.minimum(min_distances, cal_dis(pts[indices[i]], pts))
        return indices

    indices = farthest_point_sampling(pcd_points, 1024)
    pcd.points = o3d.utility.Vector3dVector([pcd_points[i] for i in indices])

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=dbscan_eps, max_nn=10))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    norm = np.linalg.norm(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points / norm))

    if is_visualize:
        o3d.visualization.draw_geometries([pcd])

    # print(np.asarray(pcd.points)[:30, :])
    # print("======")
    # print(np.asarray(pcd.normals)[:30, :])

    return pcd

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        ):

    pcd_root = '../../point_cloud_dataset_norm_v3'

    # Initialize
    # set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    suffix = Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, _ = 64, [f'class{i}' for i in range(1000)]  # assign defaults

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if half:
        model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    for filepath, subdirs, _ in os.walk(source):
        for dir in subdirs:
            if dir != "color": continue
            filepath = filepath.replace('\\', '/')

            subject, action, seq_idx = filepath[len(source)+1:].split('/')[:3]
            subfolder = os.path.join(pcd_root, subject, action, seq_idx)

            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            
            dataset = LoadImages(os.path.join(filepath, dir), img_size=imgsz, stride=stride, auto=pt)
            # Run inference
            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
            t0 = time.time()
            for path, img, im0s, _ in dataset:
                if onnx:
                    img = img.astype('float32')
                else:
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                img = img / 255.0  # 0 - 255 to 0.0 - 1.0
                if len(img.shape) == 3:
                    img = img[None]  # expand for batch dim

                # Inference
                pred = model(img, augment=augment)[0]

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1)[0]

                # Process predictions
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(pred):
                    # Rescale boxes from img_size to im0 size
                    pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).round()

                    # Write results
                    for *xyxy, _, _ in reversed(pred):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        
                        # im0 = plot_one_box(xyxy, im0s, label='Hand', color=colors(0, True), line_width=2)
                        # cv2.imshow("img", im0)
                        # cv2.waitKey(0)

                        depth_path = path.replace('color', 'depth')
                        depth_path = depth_path.replace('jpeg', 'png')

                        depth_path = depth_path.replace('\\', '/')
                        file_name = depth_path.split('/')[-1][:-4]

                        pcd_filepath = os.path.join(pcd_root, subject, action, seq_idx, file_name + '.ply')
                        if not os.path.exists(pcd_filepath):
                            open(pcd_filepath, 'w').close()
                        
                        # print(pcd_filepath)

                        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                        img_height, img_width = depth_img.shape

                        x_center_norm = xywh[0]
                        y_center_norm = xywh[1]
                        x_width_norm = xywh[2]
                        y_height_norm = xywh[3]

                        center = (img_width * x_center_norm, img_height * y_center_norm)
                        start_point = (center[0] - img_width * x_width_norm / 2, center[1] - img_height * y_height_norm / 2)
                        end_point = (center[0] + img_width * x_width_norm / 2, center[1] + img_height * y_height_norm / 2)

                        crop_depth = depth_img[int(start_point[1]):int(end_point[1]), int(start_point[0]):int(end_point[0])].copy()

                        pcd = generate_point_cloud_from_depth(crop_depth, img_width, img_height, is_visualize=False)

                        if pcd != None:
                            o3d.io.write_point_cloud(pcd_filepath, pcd)
                        else:
                            print(pcd_filepath)

            print(f'Done. ({time.time() - t0:.3f}s) :: ' + os.path.join(filepath, dir))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
