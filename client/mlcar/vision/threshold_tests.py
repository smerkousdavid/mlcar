import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def cal_undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist, mtx, dist


def collect_callibration_points():
    objpoints = []
    imgpoints = []

    images = glob.glob('./camera_cal/calibration*.jpg')
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for fname in images:
        img = mpimg.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    return imgpoints, objpoints


def compare_images(image1, image2, image1_exp="Image 1", image2_exp="Image 2", src=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=50)
    print(src[0][0])
    if src is not None:
        ax1.plot([src[0][0], src[1][0]], [src[0][1], src[1][1]], color='r', linewidth="5")
        ax1.plot([src[1][0], src[2][0]], [src[1][1], src[2][1]], color='r', linewidth="5")
        ax1.plot([src[2][0], src[3][0]], [src[2][1], src[3][1]], color='r', linewidth="5")
        ax1.plot([src[3][0], src[0][0]], [src[3][1], src[0][1]], color='r', linewidth="5")
    ax2.imshow(image2)
    ax2.set_title(image2_exp, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    isX = True if orient == 'x' else False
    sobel = cv2.Sobel(gray, cv2.CV_64F, isX, not isX)
    print(image.shape)
    print(sobel.shape)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    print(scaled_sobel.shape)
    print(np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    print("shape", sobelx.shape)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    return dir_binary


def apply_thresholds(image, ksize=11):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(40, 150))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(5, 150))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(60, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.5, 0.8))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def apply_color_thresh(img, thresh=(0.3, 1.0)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsv[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary


def apply_roi(image, x, y, width, height):
    return image[int(y):int(y + height), int(x):int(x + width)]


def combine_threshold(s_binary, combined):
    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) & (combined == 1)] = 1

    return combined_binary


def gs(i_n, scalar):
    return int(i_n * (float(scalar) / 100.0))


def gcoord(im_s, coord_1, coord_2):
    return [gs(im_s[1], coord_1 - 1), gs(im_s[0], coord_2 - 1)]


def warp(img):
    im_s = img.shape

    src = np.float32(
        [gcoord(im_s, 125, 60),  # Top right
         gcoord(im_s, 130, 97),  # Bottom right
         gcoord(im_s, -30, 97),  # Bottom left
         gcoord(im_s, -25, 60)])  # Top left

    dst = np.float32(
        [gcoord(im_s, 100, 0),  # Top right
         gcoord(im_s, 100, 100),  # Bottom right
         gcoord(im_s, 0, 100),  # Bottom left
         gcoord(im_s, 0, 0)])  # Top left

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    binary_warped = cv2.warpPerspective(img, M, (im_s[1], im_s[0]), flags=cv2.INTER_LINEAR)

    return binary_warped, Minv, src


def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    return histogram


def apply_canny(image):
    r_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im = cv2.threshold(r_im, 0.65, 1.0, cv2.THRESH_BINARY)
    cv2.imshow("test", im[1])
    cv2.waitKey(0)
    return im[1]  # cv2.Canny(im[1], 100, 100)


# def apply_horizontal_pairing(binary_image):


def apply_dilate(image, iter=1):
    return cv2.dilate(image, np.ones((5, 5), np.uint8), iterations=iter)


image = mpimg.imread("/home/smerkous/Downloads/Screenshot from 2018-01-17 21-50-51.png")
# image = apply_roi(image, 0, image.shape[0] * 0.25, image.shape[1], image.shape[0])
binary_thresh = apply_dilate(apply_thresholds(image), 2)
color_thresh = apply_dilate(apply_color_thresh(image), 1)
combined = combine_threshold(binary_thresh, color_thresh)
warped_img, min_v, src = warp(combined)
warped_img = apply_dilate(warped_img, 9)
compare_images(image, warped_img, "Original Image", "Warped Gradient", src)
# plt.show()

histogram = get_histogram(warped_img)
f, ax1 = plt.subplots(1, 1, figsize=(10, 8))
ax1.set_title("histogram", fontsize=50)
ax1.plot(histogram)
plt.show()

print("Done")
