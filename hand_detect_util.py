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
from torch.utils.data import dataset

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

subject_names = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]
gesture_names = ['charge_cell_phone','clean_glasses','close_juice_bottle','close_liquid_soap','close_milk','close_peanut_butter','drink_mug','flip_pages','flip_sponge', 'give_card',
'give_coin','handshake','high_five','light_candle','open_juice_bottle','open_letter','open_liquid_soap','open_milk','open_peanut_butter','open_soda_can','open_wallet','pour_juice_bottle'
'pour_liquid_soap','pour_milk','pour_wine','prick','put_salt','put_sugar','put_tea_bag','read_letter','receive_coin', 'scoop_spoon','scratch_sponge','sprinkle','squeeze_paper',
'squeeze_sponge','stir','take_letter_from_enveloppe','tear_paper','toast_wine','unfold_glasses','use_calculator','use_flash','wash_sponge','write']
#gesture_names = ['put_salt','use_calculator','take_letter_from_enveloppe','open_juice_bottle','open_juice_bottle','open_letter','open_liquid_soap','open_milk','open_peanut_butter','open_soda_can','open_wallet']

def generate_point_cloud_from_depth(depth_val,img_width,img_height,depth_threshold=550,is_visualize=False, voxel_size=3.0, dbscan_eps=8.0,use_voxel_downsample=True):
    # ===== crop all parts too far from camera =====
    depth_val[depth_val>depth_threshold] = 0

    if is_visualize:
        vis_depth = depth_val / 256.0
        cv2.imshow("Depth image", vis_depth)
        cv2.waitKey(0)

    # ===== convert numpy array to open3d image =====
    image = o3d.geometry.Image(depth_val.astype(np.uint16))
    intrinsic_mat = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, 475.065948, 475.065857, 315.944855, 245.287079)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(image, intrinsic_mat, depth_scale=1.0, project_valid_depth_only=True)
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    if use_voxel_downsample:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # if is_visualize:
    #     o3d.visualization.draw_geometries([pcd])

    pcd_points = np.asarray(pcd.points)
    if len(pcd_points) == 0:
        print("Not enough point")
        return (None, None)

    pcd_points *= np.array([-1, -1 , 1])

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=10, print_progress=True))

    labels = DBSCAN(eps=dbscan_eps,min_samples=10,algorithm='kd_tree').fit(pcd_points).labels_

    if len(labels) == 0:
        print("Cannot cluster point cloud")
        return (None, None)

    # if is_visualize:
    #     max_label = labels.max()
    #     print(f"point cloud has {max_label + 1} clusters")
    #     colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    #     colors[labels < 0] = 0
    #     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #     o3d.visualization.draw_geometries([pcd])

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

        # if is_visualize:
        #     o3d.visualization.draw_geometries([pcd])

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

    # if len(pcd_points) < 1024:
    #     obb = pcd.get_oriented_bounding_box()
    #     center = obb.get_center()
    #     max_bound = obb.get_max_bound()
    #     min_bound = obb.get_min_bound()
    #     radius = ((max_bound - min_bound) / 2.0) * 0.5

    #     rd_points = 2 * radius * np.random.random_sample((1024 - len(pcd_points), 3)) + (center - radius)
    #     pcd_points = np.vstack((pcd_points, rd_points))
    #     pcd.points = o3d.utility.Vector3dVector(pcd_points)

    #     # o3d.visualization.draw_geometries([pcd])
    # else:
    indices = farthest_point_sampling(pcd_points, 1024)
    pcd.points = o3d.utility.Vector3dVector([pcd_points[i] for i in indices])

    # pcd.transform(np.linalg.inv(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])))

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=dbscan_eps, max_nn=10))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    # o3d.visualization.draw_geometries([pcd])
    # print(np.asarray(pcd.points)[:10, :])
    # print("======")
    # print(np.asarray(pcd.normals)[:10, :])
    # print("######")

    # normalize by obb
    # obb = pcd.get_oriented_bounding_box()
    # rotate_mat_transpose = obb.R.transpose()
    # pcd.rotate(rotate_mat_transpose, obb.get_center())

    # if is_visualize:
    #     o3d.visualization.draw_geometries([pcd])

    # print(np.asarray(pcd.points)[:10, :])
    # print("======")
    # print(np.asarray(pcd.normals)[:10, :])
    # print("######")

    # norm = np.linalg.norm(pcd.points)
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points / norm))

    if is_visualize:
        o3d.visualization.draw_geometries([pcd])

    return (pcd, pcd.get_oriented_bounding_box())

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
        save_dir='../../processed',
        override=False
        ):

    # pcd_save_dir = '../../point_cloud_dataset_norm_v3'

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

    video_dir = os.path.join(source, 'Video_files')
    hand_annotation = os.path.join(source, 'Hand_pose_annotation_v1')

    for subject in subject_names:
        time1 = time.time()
        for gesture in gesture_names:
            if not os.path.exists(os.path.join(video_dir, subject, gesture)): continue
            for seq_idx in os.listdir(os.path.join(video_dir, subject, gesture)):
                if not os.path.exists(os.path.join(video_dir, subject, gesture, seq_idx)): continue
                # save files
                save_seq_path = os.path.join(save_dir, subject, gesture, seq_idx)
                if not override and os.path.exists(save_seq_path): continue

                if not os.path.exists(save_seq_path):
                    os.makedirs(save_seq_path)

                # read ground truth joint
                gt_ws = np.loadtxt(os.path.join(hand_annotation, subject, gesture, seq_idx, 'skeleton.txt')).astype(np.float32)
                gt_ws = gt_ws[:,1:]

                frame_num = gt_ws.shape[0]
                
                points = np.zeros((frame_num, 1024, 6)).astype(np.float32) # xyz + norm for each point
                volume_rotate = np.zeros((frame_num, 3, 3)).astype(np.float32) # rotation matrix
                bound_obb = np.zeros((frame_num, 2, 3)).astype(np.float32) # min bound & max bound of rotation matrix
                gt_xyz = np.zeros((frame_num, 63)).astype(np.float32)
                valid = [False for _ in range(frame_num)]

                image_dir = os.path.join(video_dir, subject, gesture, seq_idx, 'color')

                try:
                    dataset = LoadImages(image_dir, img_size=imgsz, stride=stride, auto=pt)
                    if pt and device.type != 'cpu':
                        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

                    t0 = time.time()
                    for path, img, im0s, _ in dataset:
                        img = torch.from_numpy(img).to(device)
                        img = img.half() if half else img.float()  # uint8 to fp16/32
                        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
                        if len(img.shape) == 3:
                            img = img[None]  # expand for batch dim

                        # Inference
                        pred = model(img, augment=augment)[0]
                        # NMS
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1)[0]
                        if len(pred):
                            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).round()
                            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            for *xyxy, _, _ in reversed(pred):
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # im0 = plot_one_box(xyxy, im0s, label='Hand', color=colors(0, True), line_width=2)
                                # cv2.imshow("img", im0)
                                # cv2.waitKey(0)

                                depth_path = path.replace('color', 'depth')
                                depth_path = depth_path.replace('jpeg', 'png')

                                depth_path = depth_path.replace('\\', '/')

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

                                pcd, obb = generate_point_cloud_from_depth(crop_depth, img_width, img_height)

                                if pcd != None:
                                    file_name = depth_path.split('/')[-1][:-4]
                                    frame_idx = int(file_name.split('.')[0].split('_')[1])

                                    valid[frame_idx] = True
                                    volume_rotate[frame_idx] = obb.R

                                    min_bound = obb.get_min_bound()
                                    max_bound = obb.get_max_bound()
                                    bound_obb[frame_idx] = np.asarray([min_bound, max_bound])

                                    # rotate & normalize point cloud
                                    pcd.rotate(obb.R.transpose(), obb.get_center())
                                    pcd_points = np.asarray(pcd.points)
                                    obb_len = (max_bound - min_bound)
                                    pcd_points = (pcd_points - min_bound) / obb_len
                                    points[frame_idx] = np.concatenate((pcd_points, np.asarray(pcd.normals)), axis=1)

                                    # rotate & normalize ground truth
                                    i_gt_xyz = gt_ws[frame_idx].reshape((21, 3))
                                    i_gt_xyz = np.matmul(i_gt_xyz, obb.R.transpose())
                                    i_gt_xyz = (i_gt_xyz - min_bound) / obb_len
                                    gt_xyz[frame_idx] = i_gt_xyz.flatten()
                except Exception as e:
                    print(e)
                
                np.save(os.path.join(save_seq_path, 'points.npy'), points)
                np.save(os.path.join(save_seq_path, 'volume_rotate.npy'), volume_rotate)
                np.save(os.path.join(save_seq_path, 'bound_obb.npy'), bound_obb)
                np.save(os.path.join(save_seq_path, 'gt_xyz.npy'), gt_xyz)
                np.save(os.path.join(save_seq_path, 'valid.npy'), valid)

        print('Done for {} - {} in {}s'.format(subject, gesture, time.time() - time1))

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
    parser.add_argument('--save_dir', default='../../processed', help='save directory')
    parser.add_argument('--override', type=bool, default=False)
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
