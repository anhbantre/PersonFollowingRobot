#!/usr/bin/env python3
    
import rospy
import ros_numpy
import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg

import math
import time
import cv2
import copy
import math as m
import numpy as np
from math import *
import open3d as o3d
# import pyrealsense2 as rs
import pyrealsense2.pyrealsense2 as rs


def init_camera():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':

            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    time.sleep(0.5)

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Processing blocks
    pc = rs.pointcloud()

    return pipeline, align, pc

def cropping_floor_old(pcd):
    """
    Return: floor point cloud, index of floor
    """

    # Downsample the point cloud with a voxel of 0.05
    pcd = pcd.voxel_down_sample(voxel_size=0.03)

    # -- Rotation
    # R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
    # pcd.rotate(R, center=(0,0,0))

    # Recompute the normal of the downsampled point cloud
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    # pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamKNN( knn=30))
    pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamRadius(radius=0.1))
    
    # Find point clouds have normal vector parallel with x axis
    nor_vector = np.asarray(pcd.normals)
    orient_x = pcd.select_by_index(np.where((0.9 < nor_vector[:, 0]) & (nor_vector[:, 0] < 1))[0])
    # orient_x.paint_uniform_color([1.0, 0, 0])


    # Plane segmentation: find the plane with the largest support in the point cloud
    plane_model, inliers = orient_x.segment_plane(distance_threshold = 0.01,
                                                ransac_n = 3,
                                                num_iterations = 100)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    floor = orient_x.select_by_index(inliers)
    # floor.paint_uniform_color([1.0, 1.0, 0])
    # outlier_cloud = orient_x.select_by_index(inliers, invert=True)
    
    return floor, inliers

def get_plane_floor(pcd):
    # Downsample the point cloud with a voxel of 0.05
    pcd = pcd.voxel_down_sample(voxel_size=0.03)

    # -- Rotation
    # R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
    # pcd.rotate(R, center=(0,0,0))

    # Recompute the normal of the downsampled point cloud
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    
    # Find point clouds have normal vector parallel with x axis
    nor_vector = np.asarray(pcd.normals)
    orient_x = pcd.select_by_index(np.where((0.9 < nor_vector[:, 0]) & (nor_vector[:, 0] < 1))[0])
    # orient_x.paint_uniform_color([0, 1, 0])


    # Plane segmentation: find the plane with the largest support in the point cloud
    plane_model, inliers = orient_x.segment_plane(distance_threshold = 0.01,
                                                ransac_n = 3,
                                                num_iterations = 200)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    floor = orient_x.select_by_index(inliers)
    # floor.paint_uniform_color([1.0, 1.0, 0])
    # outlier_cloud = orient_x.select_by_index(inliers, invert=True)
    return floor, plane_model

def cropping_floor(pcd, plane_model):
    """
    Return: floor point cloud, index of floor
    """
    [a, b, c, d] = plane_model
    # t20 = time.time()
    # Downsample the point cloud with a voxel of 0.05
    pcd = pcd.voxel_down_sample(voxel_size=0.03)

    # -- Rotation
    # R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
    # pcd.rotate(R, center=(0,0,0))

    # Recompute the normal of the downsampled point cloud
    # pcd.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    
    # Find point clouds have normal vector parallel with x axis
    # nor_vector = np.asarray(pcd.normals)
    # orient_x = pcd.select_by_index(np.where((0.9 < nor_vector[:, 0]) & (nor_vector[:, 0] < 1))[0])
    orient_x_arr = np.asarray(pcd.points).astype(np.float64)
    # print('t20: ', round(time.time() - t20, 3))
    # t21 = time.time()
    x = orient_x_arr[:, 0]
    y = orient_x_arr[:, 1]
    z = orient_x_arr[:, 2]
    # ind_floor = []
    # for i in range(orient_x_arr.shape[0]):
    #     if -0.01 < x[i]*a + y[i]*b + z[i]*c + d < 0.01:
    #         ind_floor.append(i)
    ind_floor = np.where((-0.01 < x*a + y*b + z*c + d) & (x*a + y*b + z*c + d < 0.01))[0]
    floor = pcd.select_by_index(ind_floor)
    # print('t21: ', round(time.time() - t21, 3))

    return floor

def Pointcloud2Img(pcd, image, color = (0,0,255)):
    # calculate x,y coordinate
    plane = np.asarray(pcd.points)
    N = plane.shape[0]
    # print(N)
    xy = np.empty((N,2),dtype=np.int16)
    # print(plane.shape)
    for i in range(N):
        # xy[i,0] = int(plane[i,0]*379.604/plane[i,2] + 316.587)
        # xy[i,1] = int(plane[i,1]*378.745/plane[i,2] + 244.279)
        xy[i,0] = int(plane[i,0]*211.383 / plane[i,2] + 209.629)
        xy[i,1] = int(plane[i,1]*210.831 / plane[i,2] + 120.63)
        # [ 640x480  p[316.587 244.279]  f[379.604 378.745]  Inverse Brown Conrady [-0.0558476 0.0686906 -0.00036953 -0.000512463 -0.0212583] ]
        # [ 424x240  p[209.629 120.63]  f[211.383 210.831]  Inverse Brown Conrady [-0.0548148 0.0627766 -0.000968619 0.000417071 -0.0198813] ]

        # xy[i,0] = xy[i,0] if xy[i,0] < 640 else 639
        # xy[i,1] = xy[i,1] if xy[i,1] < 480 else 479

    for i in range(xy.shape[0]):
        image[xy[i,0], image.shape[1] - xy[i,1]] = color
        # image[xy[i,1], xy[i,0]] = color
    
    return image

def add_image(img1, img2):
    """Return an img1 which putted img2 into"""
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows, 0:cols] = dst

    return img1 

def get_edge(img):
    kernel = np.ones((3,3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (9,9), 0)
    imgCanny = cv2.Canny(imgBlur, 100, 100)
    imgCanny = cv2.dilate(imgCanny, kernel, iterations=1)

    return imgCanny

def get_contour(img):
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)     # only Gray or Canny
    for cnt in contours:
        # find area of contour
        area = cv2.contourArea(cnt)
        cv2.fillPoly(img, pts =[cnt], color=(0,0,0))
        # if area < 7000:
        #     # cv2.drawContours(img, cnt, -1, (0, 0, 0), 3)
        #     cv2.fillPoly(img, pts =[cnt], color=(0,0,0))
        if area > 500:
            epsilon = 0.005*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            cv2.fillPoly(img, pts =[approx], color=(255,255,255))
    
    return img

def calculate_delta(img):
    line1 = np.array(img[img.shape[0] - 70, :])
    line2 = np.array(img[img.shape[0] - 160, :])
    ind1 = np.where(line1 > 0)
    ind2 = np.where(line2 > 0)
    img[img.shape[0] - 70, :] = 255
    img[img.shape[0] - 160, :] = 255
    lb_edge = ind1[0][0]
    rb_edge = ind1[0][-1]
    lt_edge = ind2[0][0]
    rt_edge = ind2[0][-1]

    mode = 1        # 0: khong gian rong, 1: khong gian hep
    # line angle = atan(424/30) = 85 deg
    thres_edge = 30
    if (0 <= lb_edge <= thres_edge) and (img.shape[1]-thres_edge <= rb_edge <= img.shape[1]) \
        and (0 <= lt_edge <= thres_edge) and (img.shape[1]-thres_edge <= rt_edge <= img.shape[1]):
        mode = 0
    else:
        mode = 1

    xM = (lt_edge + rt_edge) // 2
    delta = xM - (img.shape[1]//2)

    mode_pub.publish(mode)
    delta_pub.publish(delta)
    # print(lb_edge, rb_edge, lt_edge, rt_edge)
    print('mode: ', mode)
    print('delta:', delta)
        
    return delta, xM

def get_line(img):
    # Probabilistic Line Transform
    img = cv2.Canny(img, 100, 150)

    linesP = cv2.HoughLinesP(img, 1, np.pi / 180, 10, None, 50, 50)
    img[img.shape[0] - 70, :] = 255
    img[img.shape[0] - 170, :] = 255

    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            angle = int(math.atan((l[1]-l[3]) / (l[2]-l[0])) * 180/math.pi)
            if 40 < angle < 140:
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)

    #  Standard Hough Line Transform
    # lines = cv2.HoughLines(img, 1, np.pi / 180, 150, None, 0, 0)
    # # Draw the lines
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv2.line(img, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)
    
    return img

def crop_pcl(pcl, Z_range):
    # pcl = np.asarray(point_cloud.points)
    # mask_x = (pcl[:,0]>X_range[0])&(pcl[:,0]<X_range[1])
    # mask_y = (pcl[:,1]>Y_range[0])&(pcl[:,1]<Y_range[1])
    mask_z = (pcl[:,2]>Z_range[0])&(pcl[:,2]<Z_range[1])
    mask = mask_z
    pcl_filtered = pcl[mask]
    return pcl_filtered

def bb_callback(data):
    global depth_frame, color_image_raw, color_image
    
    color_image_raw[color_image.shape[1] - int(data.x), int(data.y)] = [0, 255, 0]
    d = depth_frame.get_distance(int(data.y), color_image.shape[1]- int(data.x))
    d_pub.publish(d)

def stt_callback(data):
    global STATUS
    
    STATUS = data.data

if __name__ == "__main__":
    pipeline, align, pc = init_camera()
    pcd = o3d.geometry.PointCloud()
    iteration = 0
    STATUS = 1
    floor_plane = None 
    depth_frame = None
    color_image_raw = None 
    color_image = None
    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    mode_pub = rospy.Publisher('/mode', std_msgs.msg.Int16, queue_size=10)
    delta_pub = rospy.Publisher('/delta', std_msgs.msg.Int16, queue_size=10)
    img_pub = rospy.Publisher('/color_image', sensor_msgs.msg.Image, queue_size=10)
    d_pub = rospy.Publisher('/distance', std_msgs.msg.Float32, queue_size=10)
    center_bb_sub = rospy.Subscriber('/center_boundingbox', geometry_msgs.msg.Point, bb_callback)
    status = rospy.Subscriber('/status', std_msgs.msg.Int16, stt_callback)
    rospy.init_node('process_image', anonymous=True)

    while True:
        if STATUS == 0:
            try:
                print('i = ', iteration)
                start_time = time.time()
                t0 = time.time()
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not depth_frame or not color_frame:
                    continue

                # Grab new intrinsics (may be changed by decimation)
                # depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
                # color_intrinsics = rs.video_stream_profile(color_frame.profile).get_intrinsics()
                # w, h = depth_intrinsics.width, depth_intrinsics.height
                # print(color_intrinsics)

                depth_image_raw = np.asanyarray(depth_frame.get_data())
                color_image_raw = np.asanyarray(color_frame.get_data())

                # Rotate image
                depth_image = cv2.rotate(depth_image_raw, cv2.ROTATE_90_CLOCKWISE)
                color_image = cv2.rotate(color_image_raw, cv2.ROTATE_90_CLOCKWISE)
                w, h = color_image.shape[1], color_image.shape[0]
                original_image = color_image.copy()

                # Public color image
                ros_img = ros_numpy.msgify(sensor_msgs.msg.Image, color_image, encoding='bgr8')     # convert OpenCV image to ROS image msg
                img_pub.publish(ros_img)
                
                pc.map_to(color_frame)
                points = pc.calculate(depth_frame)
                t0 = time.time() - t0
                print(f'Time to get image: {round(t0, 2)}s')

                t1 = time.time()
                # Pointcloud data to arrays
                v = points.get_vertices()
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3).astype(np.float64)  # xyz
                # texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
                verts_filter = crop_pcl(verts, [0, 4])

                # np.set_printoptions(threshold=100000)

                # Convert numpy to open3d point cloud 
                pcd.points = o3d.utility.Vector3dVector(verts_filter)
                t1 = time.time() - t1
                print(f'Time to convert to open3d data: {round(t1, 2)}s')

                t2 = time.time()
                if iteration == 0 or iteration % 20 == 0:
                    floor, floor_plane = get_plane_floor(pcd)
                else:
                    floor = cropping_floor(pcd, floor_plane)
                # floor, ind_floor = cropping_floor_old(pcd)
                t2 = time.time() - t2
                print(f'Time to crop floor: {round(t2, 2)}s')
                
                t3 = time.time()
                blank_image = np.zeros((h, w, 3), np.uint8)
                mask = Pointcloud2Img(floor, blank_image)
                t3 = time.time() - t3
                print(f'Time to calculate in pixel: {round(t3, 2)}s')

                t4 = time.time()
                # preprocess image
                kernel = np.ones((5,5), np.uint8)
                dilation_image = cv2.dilate(mask, kernel, iterations=3)
                dilation_image[:, 0:5]       = [0,0,0]
                dilation_image[:, (w-5):(w)] = [0,0,0]
                dilation_image[0:5, :]       = [0,0,0]
                dilation_image[(h-5):(h), :] = [0,0,0]
                processed_image = add_image(color_image, dilation_image)

                edge_image = get_edge(dilation_image)
                mask_image = get_contour(edge_image)
                delta, xM = calculate_delta(mask_image)
                # result_image = get_line(mask_image)
                # cv2.line(result_image, (xM, (h-50)), (w//2, (h-50)), (255,255,255), 3, cv2.LINE_AA)
                t4 = time.time() - t4
                print(f'Time to preprocessing image: {round(t4, 2)}s')

                # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
                # processed_image = add_image(color_image, mask_image)

                # o3d.visualization.draw_geometries([pcd, floor] + [axis_pcd])
                # Show image
                # cv2.imshow('color image', original_image)
                # cv2.imshow('result image', color_image_raw)
                # key = cv2.waitKey(30)
                # if key == ord("q"):
                #     break

                # Save image
                # cv2.imwrite(f'/home/huyan/catkin_ws/src/vision/src/data/video3/image/image_{i}.png', original_image)
                # cv2.imwrite(f'/home/huyan/catkin_ws/src/vision/src/data/video5/processed/processed_image_{i}.png', mask_image)
                # Save point cloud
                # o3d.io.write_point_cloud(f'/home/huyan/catkin_ws/src/vision/src/data/video3/pc/point_cloud_{i}.ply', pcd)
                
                print(f'Total time: {round(time.time() - start_time, 2)}s\n')
                iteration += 1

            except Exception as e:
                print(e)
                pass

    rospy.spin()
    # Stop streaming
    pipeline.stop()