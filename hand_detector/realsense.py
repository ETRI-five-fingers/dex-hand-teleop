import os.path

from utils.utils import AttrDict
import numpy as np
import cv2

import pyrealsense2 as rs


class RealSenseBase:
    def __init__(self):
        """
        Currently, each cam is used as following:
            [front] cam is used for recording evaluation process
            [rear] cam is used for observing the environment
        """
        self.cam_id = AttrDict(front='818312070927', rear='')

    @staticmethod
    def check_devices():
        ctx = rs.context()
        devices = list(ctx.query_devices())
        print(devices)
        return devices


class RealSense(RealSenseBase):
    def __init__(self, args=None):
        super().__init__()
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.cam_id.front)
        self.args = args if args is not None else AttrDict(width=640, height=480, fps=30)

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            raise ConnectionError

        self.config.enable_stream(rs.stream.depth, self.args.width, self.args.height, rs.format.z16, self.args.fps)

        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, self.args.fps)
        else:
            self.config.enable_stream(rs.stream.color, self.args.width, self.args.height, rs.format.bgr8, self.args.fps)

        # Start streaming
        self.pipeline.start(self.config)

    def display_info(self):
        depth, color = self.get_np_images()
        if depth is None or color is None:
            raise ValueError

        print("====================")
        print("-Vision Information-")
        print("====================")
        print("* Image Stream")
        print("    Width: {}, Height: {}, FPS: {}".format(self.args.width, self.args.height, self.args.fps))
        print("* Actual Data")
        print("    Depth shape: {}, min / max: {} / {}, dtype: {}".format(depth.shape, depth.min(), depth.max(),
                                                                          depth.dtype))
        print("    Color shape: {}, min / max: {} / {}, dtype: {}".format(color.shape, color.min(), color.max(),
                                                                          color.dtype))

    def stop_stream(self):
        self.pipeline.stop()

    def get_np_images(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())  # (h, w, 1)
        color_image = np.asanyarray(color_frame.get_data())  # (h, w, 3)
        return depth_image, color_image


from threading import Event


class RealSenseApp(RealSenseBase):
    def __init__(self, file: str = "", args=None):
        super().__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.cam_id.front)
        self.args = args if args is not None else AttrDict(width=640, height=480, fps=30)

        device_product_line = self.connect_to_device()

        self.config.enable_stream(rs.stream.depth, self.args.width, self.args.height, rs.format.z16, self.args.fps)

        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, self.args.fps)
        else:
            self.config.enable_stream(rs.stream.color, self.args.width, self.args.height, rs.format.bgr8, self.args.fps)

        # Start streaming
        self.pipeline.start(self.config)

        self.rgb_video_file = file
        self.event = Event()
        self.session = None

        self.rgb_stream = None
        self.depth_stream = None
        self.read_count = 0

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()    # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print("Stream stopped")

    def connect_to_device(self, dev_idx=0):
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            raise ConnectionError

        return device_product_line

        # # TODO,
        # if not os.path.exists(self.rgb_video_file):
        #     pass
        # else:
        #     pass

    @staticmethod
    def _get_intrinsic_mat_from_coeffs(coeffs):
        return np.array([[coeffs.fx, 0, coeffs.ppx],
                         [0, coeffs.fy, coeffs.ppy],
                         [0, 0, 1]])

    @property
    def camera_intrinsics(self):
        profile = self.pipeline.get_active_profile()

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()

        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        # color_intrinsics or depth_intrinsics?
        return self._get_intrinsic_mat_from_coeffs(depth_intrinsics)

    def fetch_rgb_and_depth(self):
        if not self.rgb_video_file:
            # self.event.wait(0.1)
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_image = np.asanyarray(depth_frame.get_data())  # (h, w, 1)
            color_image = np.asanyarray(color_frame.get_data())  # (h, w, 3)

            # depth_image = np.transpose(depth_image, [1, 0])
            # color_image = np.transpose(color_image, [1, 0, 2])

            # print("depth shape: ", depth_image.shape)
            # print("color shape: ", color_image.shape)

            # is_true_depth = depth_image.shape[0] == 480
            # if is_true_depth:
            #     depth_image = cv2.flip(depth_image, 1)
            #     color_image = cv2.flip(color_image, 1)

            # print("after depth shape: ", depth_image.shape)
            # print("after color shape: ", color_image.shape)

            # depth scaling
            depth_image = (depth_image * self.depth_scale).astype(np.float32)
            return cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_image
        else:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None, None

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())  # (h, w, 1)
            color_image = np.asanyarray(color_frame.get_data())  # (h, w, 3)
            return cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_image

    def record(self, rgb_filename):
        raise NotImplementedError


if __name__ == "__main__":
    # RealSenseApp
    cam = RealSenseApp()
    name_of_window = 'SN: ' + str(cam.cam_id)

    intr = cam.camera_intrinsics
    print(intr)

    while True:
        color, depth = cam.fetch_rgb_and_depth()
        bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # Display images
        cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name_of_window, bgr)
        cv2.imshow("depth", depth)
        print("min/max: {}/{}".format(depth.min(), depth.max()))
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print(f"User pressed break key for SN: {cam.cam_id}")
            break

    # # RealSense
    # cam = RealSense()
    # name_of_window = 'SN: ' + str(cam.cam_id)
    #
    # while True:
    #     depth, color = cam.get_np_images()
    #
    #     # Display images
    #     cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow(name_of_window, color)
    #     key = cv2.waitKey(1)
    #     # Press esc or 'q' to close the image window
    #     if key & 0xFF == ord('q') or key == 27:
    #         print(f"User pressed break key for SN: {cam.cam_id}")
    #         break
