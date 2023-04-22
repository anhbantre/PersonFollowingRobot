#!/usr/bin/env python3
    
import rospy
import ros_numpy
import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg

import time
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2.pyrealsense2 as rs
from math import *


HEIGHT_FRAME = 240
WEIGHT_FRAME = 424
FPS = 30
# Intrinsic parameters of the camera
FOCAL_DIST_X = 211.383
FOCAL_DIST_Y = 210.831
P_X = 209.629
P_Y = 120.63

def initialize_camera():
    """
    Initialize camera and return pipeline, align, and pointcloud objects
    """
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Create a pipeline wrapper and resolve it to get a pipeline profile
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # Check if RGB camera sensor is available, exit if not
    rgb_sensor_found = False
    for sensor in device.sensors:
        if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
            rgb_sensor_found = True
            break

    if not rgb_sensor_found:
        print("The robot requires Depth camera with Color sensor")
        exit(0)
    
    # Enable depth and color streams with specified settings
    config.enable_stream(rs.stream.depth, WEIGHT_FRAME, HEIGHT_FRAME, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WEIGHT_FRAME, HEIGHT_FRAME, rs.format.bgr8, FPS)

    # Start streaming
    pipeline.start(config)
    time.sleep(0.5)         # Sleep to allow camera to fully initialize

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    width, height = depth_intrinsics.width, depth_intrinsics.height

    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Processing blocks
    pointcloud  = rs.pointcloud()

    return pipeline, align, pointcloud 

def find_floor_plane(point_cloud: o3d.geometry.PointCloud) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Returns the plane which contains the floor point cloud

    Args:
        point_cloud (o3d.geometry.PointCloud): The original point cloud

    Returns:
        floor (o3d.geometry.PointCloud): The point cloud of the floor
        floor_index (np.ndarray): The index of the floor point cloud
    """
    # Downsample the point cloud with a voxel of 0.03
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size=0.03)

    # Recompute the normals of the downsampled point cloud
    downsampled_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    
    # Find point clouds having normal vector parallel to the x-axis
    normals = np.asarray(downsampled_pcd.normals)
    x_axis_normals = downsampled_pcd.select_by_index(np.where((0.9 < normals[:, 0]) & (normals[:, 0] < 1))[0])

    # Segment plane: find the plane with the largest support in the point cloud
    plane_model, inliers = x_axis_normals.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=200)
    # [a, b, c, d] = plane_model
    # Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0

    floor_plane = x_axis_normals.select_by_index(inliers)
    return floor_plane, plane_model

def crop_floor(point_cloud: o3d.geometry.PointCloud,
                plane_model: np.ndarray) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Return the floor point cloud and its index

    Args:
        point_cloud (o3d.geometry.PointCloud): The original point cloud
        plane_model (np.ndarray): The model of the floor plane

    Returns:
        floor (o3d.geometry.PointCloud): The point cloud of the floor
        floor_index (np.ndarray): The index of the floor point cloud
    """
    # Extract the coefficients of the floor plane model
    a, b, c, d = plane_model

    # Downsample the point cloud with a voxel of 0.03
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.03)
    
    # Find the points have the normal vector parallel with x axis
    x_axis_arr = np.asarray(pcd.points).astype(np.float64)
    x = x_axis_arr[:, 0]
    y = x_axis_arr[:, 1]
    z = x_axis_arr[:, 2]
    floor_index = np.where((-0.01 < x*a + y*b + z*c + d) & (x*a + y*b + z*c + d < 0.01))[0]
    floor = point_cloud.select_by_index(floor_index)

    return floor, floor_index

def point_cloud_to_image(point_cloud: o3d.geometry.PointCloud,
                        image: np.ndarray,
                        color=(0, 0, 255),
                        focal_dist_x=211.383,
                        focal_dist_y=210.831,
                        p_x=209.629, p_y=120.63) -> np.ndarray:
    """
    Convert point cloud to image

    Args:
        point_cloud (o3d.geometry.PointCloud): The original point cloud
        image (np.ndarray): The RGB image to store the point cloud, shape (height, width, 3)
        color (tuple): The color of the point cloud. Defaults to (0, 0, 255)
        focal_dist_x (float): The focal distance along the x-axis of the camera. Defaults to 211.383
        focal_dist_y (float): The focal distance along the y-axis of the camera. Defaults to 210.831
        p_x (float): The x-coordinate of the principal point of the camera. Defaults to 209.629
        p_y (float): The y-coordinate of the principal point of the camera. Defaults to 120.63

    Returns:
        image (np.ndarray): The RGB image with the projected points colored
    """
    # Project each point onto the image plane
    plane = np.asarray(pcd.points)
    N = plane.shape[0]
    img_points = np.empty((N, 2), dtype=np.int16)

    for i in range(N):
        img_points[i, 0] = int(point_cloud[i, 0] * focal_dist_x / point_cloud[i, 2] + p_x)
        img_points[i, 1] = int(point_cloud[i, 1] * focal_dist_y / point_cloud[i, 2] + p_y)
        # [ 424x240  p[209.629 120.63]  f[211.383 210.831]  Inverse Brown Conrady [-0.0548148 0.0627766 -0.000968619 0.000417071 -0.0198813] ]

    # Color each projected point in the image
    for i in range(img_points.shape[0]):
        image[img_points[i, 0], image.shape[1] - img_points[i, 1]] = color
    
    return image

def add_image(background: np.ndarray, logo: np.ndarray) -> np.ndarray:
    """
    Create a new image by adding a logo to a background image

    Args:
        background (np.ndarray): The 3-channel image to use as the background
        logo (np.ndarray): The 3-channel image to use as the logo
    
    Returns:
        background (np.ndarray): A new image with the logo added to the image
    """
    bg_rows, bg_cols, bg_channels = background.shape
    logo_rows, logo_cols, logo_channels = logo.shape

    # Create a Region of Interest (ROI) for the logo on the background
    roi = background[:logo_rows, :logo_cols]

    # Convert the logo to grayscale and create a binary mask
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the ROI to create a black background for the logo
    bg_masked = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Apply the mask to the logo to isolate the foreground
    logo_masked = cv2.bitwise_and(logo, logo, mask=mask)

    # Combine the masked logo and ROI to create the final image
    dst = cv2.add(bg_masked, logo_masked)
    background[:logo_rows, :logo_cols] = dst

    return background 

def find_object_edge(image: np.ndarray) -> np.ndarray:
    """
    Find the edge of object in the image

    Args:
        image (np.ndarray): The image to find the edge of object in

    Returns:
        dilated_image (np.ndarray): The edge of object in the image
    """
    kernel = np.ones((3, 3), np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    canny_image = cv2.Canny(blurred_image, 100, 100)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)

    return dilated_image

def get_contour(image: np.ndarray) -> np.ndarray:
    """
    Find the contour of the lagrest object in the image

    Args:
        image (np.ndarray): The image to find the contour

    Returns:
        image (np.ndarray): The image with the contour of the largest object
    """
    # Find all contours in the image
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)     # only Gray or Canny
    
    # Loop through all contours and find the largest one
    for contour  in contours:
        # find the area of contour
        area = cv2.contourArea(contour )

        # Remove small contours
        if area < 500:
            cv2.fillPoly(image, pts=[contour], color=(0,0,0))
        else:
            # Fill the largest contour with black
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.fillPoly(image, pts=[approx], color=(255,255,255))
    
    return image

def calculate_delta(image: np.ndarray) -> np.ndarray:
    """
    Compute the delta that is the center point of the floor horizontally
    
    Args:
        image (np.ndarray): The image to compute the delta

    Returns:
        delta (float): The gap between the center of the frame and the center of the floor
        center_point (float): The center point of the floor
    """
    # Get the bottom two lines of the image
    line1 = np.array(image[image.shape[0] - 70, :])
    line2 = np.array(image[image.shape[0] - 160, :])

    # Get the indices of non-zero values in the bottom two lines
    ind1 = np.where(line1 > 0)
    ind2 = np.where(line2 > 0)

    # Replace the bottom two lines with a white line
    image[image.shape[0] - 70, :] = 255
    image[image.shape[0] - 160, :] = 255
    
    # Get the left and right edges of the bottom line and the left and right edges of the second bottom line
    lb_edge = ind1[0][0]
    rb_edge = ind1[0][-1]
    lt_edge = ind2[0][0]
    rt_edge = ind2[0][-1]

    # Determine whether the space is narrow or wide, 0: wide, 1: narrow
    # line angle = atan(424/30) = 85 deg
    space_type = 1
    threshold_edge = 30
    if (0 <= lb_edge <= threshold_edge ) and (image.shape[1]-threshold_edge  <= rb_edge <= image.shape[1]) \
        and (0 <= lt_edge <= threshold_edge ) and (image.shape[1]-threshold_edge  <= rt_edge <= image.shape[1]):
        space_type = 0     # Narrow space
    else:
        space_type = 1     # Wide space

    # Caculate the center point of two lines
    center_point = (lt_edge + rt_edge) // 2

    # Calculate the delta and publish the space type and delta
    delta = center_point - (image.shape[1]//2)
    mode_pub.publish(space_type)
    delta_pub.publish(delta)
        
    return delta, center_point

def crop_point_cloud(point_cloud: np.ndarray, Z_range: tuple) -> np.ndarray:
    """
    Crop a point cloud by fltering out points outside a given Z range to reduce the number of points
    
    Args:
        point_cloud (np.ndarray): An Nx3 array representing a point cloud, where N is the number of points
        Z_range (tuple): A tuple (z_min, z_max) representing the range of Z values to keep

    Returns:
        point_cloud_filtered (np.ndarray): An Mx3 array representing the filtered point cloud,
                                            where M is the number of points inside the Z range
    """
    # Create a boolean mask that is True for points inside the Z range
    mask_z = (point_cloud[:,2]>Z_range[0])&(point_cloud[:,2]<Z_range[1])
    
    # Apply the mask to the point cloud to filter out points outside the Z range
    mask = mask_z
    point_cloud_filtered  = point_cloud[mask]
    return point_cloud_filtered 

def bb_callback(data):
    """Callback function for caculate the distance between robot and object"""
    global depth_frame, color_image_raw, color_image
    
    # Update the pixel in the color image
    color_image_raw[color_image.shape[1] - int(data.x), int(data.y)] = [0, 255, 0]

    # Get the distance of the selected pixel from the depth frame
    d = depth_frame.get_distance(int(data.y), color_image.shape[1]- int(data.x))
    
    # Publish the distance of the selected pixel
    d_pub.publish(d)

def stt_callback(data):
    """
    Callback function for the button_status of the robot
    0: run, 1: stop
    """
    global button_status 
    
    button_status  = data.data

if __name__ == "__main__":
    pipeline, align, pc = initialize_camera()
    pcd = o3d.geometry.PointCloud()
    iteration = 0
    button_status  = 1
    floor_plane = None 
    depth_frame = None
    color_image_raw = None 
    color_image = None
    
    mode_pub = rospy.Publisher('/mode', std_msgs.msg.Int16, queue_size=10)
    delta_pub = rospy.Publisher('/delta', std_msgs.msg.Int16, queue_size=10)
    img_pub = rospy.Publisher('/color_image', sensor_msgs.msg.Image, queue_size=10)
    d_pub = rospy.Publisher('/distance', std_msgs.msg.Float32, queue_size=10)
    center_bb_sub = rospy.Subscriber('/center_boundingbox', geometry_msgs.msg.Point, bb_callback)
    button_status  = rospy.Subscriber('/button_status ', std_msgs.msg.Int16, stt_callback)
    rospy.init_node('process_image', anonymous=True)

    while True:
        if button_status == 0:
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

                # Convert pointcloud data to arrays
                t1 = time.time()
                v = points.get_vertices()
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3).astype(np.float64)  # xyz
                verts_filter = crop_point_cloud(verts, [0, 4])

                # Convert numpy to open3d point cloud 
                pcd.points = o3d.utility.Vector3dVector(verts_filter)
                t1 = time.time() - t1
                print(f'Time to convert to open3d data: {round(t1, 2)}s')

                # Each 20 iteration, crop the floor to reduce the run time
                t2 = time.time()
                if iteration == 0 or iteration % 20 == 0:
                    floor, floor_plane = find_floor_plane(pcd)
                else:
                    floor = crop_floor(pcd, floor_plane)
                t2 = time.time() - t2
                print(f'Time to crop floor: {round(t2, 2)}s')
                
                # Convert the floor point clouds to image pixels
                t3 = time.time()
                blank_image = np.zeros((h, w, 3), np.uint8)
                mask = point_cloud_to_image(point_cloud=floor, image=blank_image,
                                            focal_dist_x=FOCAL_DIST_X, focal_dist_y=FOCAL_DIST_Y,
                                            p_x=P_X, p_y=P_Y)
                t3 = time.time() - t3
                print(f'Time to calculate in pixel: {round(t3, 2)}s')
                
                # Preprocess image
                t4 = time.time()
                kernel = np.ones((5,5), np.uint8)
                dilation_image = cv2.dilate(mask, kernel, iterations=3)
                dilation_image[:, 0:5]       = [0,0,0]
                dilation_image[:, (w-5):(w)] = [0,0,0]
                dilation_image[0:5, :]       = [0,0,0]
                dilation_image[(h-5):(h), :] = [0,0,0]
                processed_image = add_image(background=color_image, logo=dilation_image)

                edge_image = find_object_edge(dilation_image)
                mask_image = get_contour(edge_image)
                delta, xM = calculate_delta(mask_image)
                t4 = time.time() - t4
                print(f'Time to preprocessing image: {round(t4, 2)}s')
                
                print(f'Total time: {round(time.time() - start_time, 2)}s\n')
                iteration += 1

            except Exception as e:
                print(e)
                pass

    rospy.spin()
    # Stop streaming
    pipeline.stop()