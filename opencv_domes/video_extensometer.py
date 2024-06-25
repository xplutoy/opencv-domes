import os
import math
import csv
import cv2

# 相机的帧率，以及拉伸机的采样频率
sample_frq = 10  # hz

# 拉伸条在图像中的位置，用于切割图像
target_h_min = 278
target_h_max = 318
target_w_min = 120

# 颜色检查的hsv范围
hsv_lower_red = (160, 60, 60)
hsv_upper_red = (180, 255, 255)

# 静息长度，用作实际标尺
g_ruler_length = 12.5  # mm

# 静息长度，用在像素标尺
g_pixel_length = 575

# debug
test_result_dir = "D:\\Codes\\opencv-domes\\opencv_domes\\result\\"


def process_per_video(filepath):

    pixel_len_list = []

    try:
        cap = cv2.VideoCapture(filepath)

        if not cap.isOpened():
            print("Error opening video stream or file")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_interval = _calc_frame_interval(fps, sample_frq)

        # 处理帧
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting...")
                break

            if i % frame_interval == 0:
                pixel_len = process_per_frame(frame, len(pixel_len_list))
                pixel_len_list.append(pixel_len)

            i = i + 1

        # 像素距离转换成实际距离
        stretch_lengths = _convert_pixel_2_length(pixel_len_list)

        # debug
        with open(os.path.join(test_result_dir, "output.csv"), "w", newline="") as file:
            writer = csv.writer(file)
            for pl, sl in zip(pixel_len_list, stretch_lengths):
                writer.writerow([pl, sl])

        return stretch_lengths

    finally:
        cap.release()


def process_per_frame(frame, idx):
    """process_per_frame _summary_
    返回图片中两个标记点之间的像素点个数。
    :param _type_ frame: _description_
    :return _type_: _description_
    """
    # crop
    im_crop = frame[target_h_min:target_h_max, target_w_min:]

    # color detect
    hsv_image = cv2.cvtColor(im_crop, cv2.COLOR_BGR2HSV)
    hsv_red_mask = cv2.inRange(hsv_image, hsv_lower_red, hsv_upper_red)

    # denoise
    hsv_mask_median = cv2.medianBlur(hsv_red_mask, 9)

    # 找到最大的两个轮廓
    contours, _ = cv2.findContours(hsv_mask_median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # debug
    cv2.imwrite(os.path.join(test_result_dir, f"{idx}_crop.png"), im_crop)
    cv2.imwrite(os.path.join(test_result_dir, f"{idx}_mask0.png"), hsv_red_mask)
    cv2.imwrite(os.path.join(test_result_dir, f"{idx}_mask1.png"), hsv_mask_median)

    return _calc_pixel_length(sorted_contours)


def _calc_frame_interval(videp_fps: int, sample_frq: int) -> int:
    return math.ceil(videp_fps / sample_frq)


def _convert_pixel_2_length(pixel_lengths):
    return [pl * g_ruler_length / g_pixel_length for pl in pixel_lengths]


def _calc_pixel_length(sorted_contours):
    """_calc_pixel_length _summary_
    如果无法得出一个合理的长度，则返回的pixel长度为0.
    :param _type_ sorted_contours: _description_
    :return _type_: _description_
    """
    if len(sorted_contours) < 2:
        pixel_len = 0
    elif cv2.contourArea(sorted_contours[0]) - cv2.contourArea(sorted_contours[1]) * 4 >= 0:
        pixel_len = 0
    else:
        # 轮廓的矩形边界
        x0, _, w0, h0 = cv2.boundingRect(sorted_contours[0])  # x,y,w,h
        x1, _, w1, h1 = cv2.boundingRect(sorted_contours[1])

        # 第二个色块不是特别精准，对其宽度进行修正
        if w1 > w0:
            w1 = w0

        if h0 / h1 > 2:
            pixel_len = 0
        else:
            pixel_len = abs(x0 + w0 / 2 - x1 - w1 / 2)

    return pixel_len


if __name__ == "__main__":
    video_filepath = "D:\\Codes\\opencv-domes\\opencv_domes\\6.avi"
    stretch_lengths = process_per_video(video_filepath)
    if stretch_lengths is not None:
        print(f"0,1,100,500 : {stretch_lengths[0]}, {stretch_lengths[1]},{stretch_lengths[100]},{stretch_lengths[500]}")
