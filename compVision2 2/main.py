import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline


def select_corresponding_points_images(image1_rgb, image2_rgb, num_points=9):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image1_rgb)
    ax[0].set_title("Image 1")
    ax[0].axis('off')

    ax[1].imshow(image2_rgb)
    ax[1].set_title("Image 2")
    ax[1].axis('off')

    points_im1 = []
    points_im2 = []

    print(f"Please select {num_points} pairs of corresponding points.")

    for i in range(num_points):
        plt.suptitle(f"Select point {i+1} in Image 1")
        plt.draw()
        x1, y1 = plt.ginput(1, timeout=0)[0]
        points_im1.append((x1, y1))
        print(f"Point {i+1} in Image 1: ({x1}, {y1})")

        plt.suptitle(f"Select point {i+1} in Image 2")
        plt.draw()
        x2, y2 = plt.ginput(1, timeout=0)[0]
        points_im2.append((x2, y2))
        print(f"Point {i+1} in Image 2: ({x2}, {y2})")

    plt.close()

    return np.array(points_im1), np.array(points_im2)

def computeH(points_im1, points_im2):
    num_points = points_im1.shape[0]
    A = []

    for i in range(num_points):
        x1, y1 = points_im1[i]
        x2, y2 = points_im2[i]
        A.append([-x1, -y1, -1,  0,   0,   0, x1*x2, y1*x2, x2])
        A.append([  0,   0,  0, -x1, -y1, -1, x1*y2, y1*y2, y2])

    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape((3, 3))
    H = H / H[2, 2]
    print(H)
    return H

def warp(image, homography):
    h, w, channels = image.shape

    corners = np.array([
        [0, 0, 1],
        [w - 1, 0, 1],
        [0, h - 1, 1],
        [w - 1, h - 1, 1]
    ]).T
    transformed_corners = homography @ corners
    transformed_corners /= transformed_corners[2, :]

    x_min, y_min = np.min(transformed_corners[0, :]), np.min(transformed_corners[1, :])
    x_max, y_max = np.max(transformed_corners[0, :]), np.max(transformed_corners[1, :])

    x_offset = int(np.floor(x_min))
    y_offset = int(np.floor(y_min))
    output_width = int(np.ceil(x_max - x_min))
    output_height = int(np.ceil(y_max - y_min))

    warped_image = np.zeros((output_height, output_width, channels), dtype=np.uint8)

    map_x, map_y = np.meshgrid(np.arange(output_width), np.arange(output_height))
    map_x = map_x + x_offset
    map_y = map_y + y_offset

    H_inv = np.linalg.inv(homography)
    src_x = (H_inv[0, 0] * map_x + H_inv[0, 1] * map_y + H_inv[0, 2]) / \
            (H_inv[2, 0] * map_x + H_inv[2, 1] * map_y + H_inv[2, 2])
    src_y = (H_inv[1, 0] * map_x + H_inv[1, 1] * map_y + H_inv[1, 2]) / \
            (H_inv[2, 0] * map_x + H_inv[2, 1] * map_y + H_inv[2, 2])

    warped_image = cv2.remap(image, src_x.astype(np.float32), src_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return warped_image, x_offset, y_offset

def add_noise_to_points(points, variance):
    noise = np.random.normal(0, np.sqrt(variance), points.shape)
    noisy_points = points + noise
    return noisy_points

def blend_images(base_image, warped_image, x_offset, y_offset):
    h_base, w_base, _ = base_image.shape
    h_warped, w_warped, _ = warped_image.shape

    x_min = min(0, x_offset)
    y_min = min(0, y_offset)
    x_max = max(w_base, x_offset + w_warped)
    y_max = max(h_base, y_offset + h_warped)

    width = int(x_max - x_min)
    height = int(y_max - y_min)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    x_base_offset = -x_min
    y_base_offset = -y_min
    canvas[y_base_offset:y_base_offset + h_base, x_base_offset:x_base_offset + w_base] = base_image

    x_warped_offset = x_offset - x_min
    y_warped_offset = y_offset - y_min

    mask_warped = np.any(warped_image > 0, axis=2)
    canvas[y_warped_offset:y_warped_offset + h_warped, x_warped_offset:x_warped_offset + w_warped][mask_warped] = warped_image[mask_warped]

    return canvas

def stitch_two_images(image1_rgb, image2_rgb):
    points_im1, points_im2 = select_corresponding_points_images(image1_rgb, image2_rgb)

    H = computeH(points_im2, points_im1)
    warped_image2, x_min, y_min = warp(image2_rgb, H)
    panorama = blend_images(image1_rgb, warped_image2, x_min, y_min)
    return panorama

#This below stitch is used for adding noise , normalization, and I also used it for pre defined points like left_1....
#In order to be used you need to also change the the stitching function also left_to_right_stitching, middle_out_stitching, first_out_then_middle_stitching

# def stitch_two_images(image1_rgb, image2_rgb, points_im1, points_im2, variance=0):
#     # Add noise if variance is specified
#     if variance > 0:
#         points_im1 = add_noise_to_points(points_im1, variance)
#         points_im2 = add_noise_to_points(points_im2, variance)

#     # Normalize the points for both images
#     # normalized_points_im1, T_im1 = normalize_points(points_im1)
#     # normalized_points_im2, T_im2 = normalize_points(points_im2)

#     # Compute the homography using the normalized points
#     H = computeH(points_im2, points_im1)
    
#     # Denormalize the homography matrix to get it in the original coordinate system
#     #H = np.linalg.inv(T_im1) @ H_normalized @ T_im2

#     # Warp the second image using the computed homography
#     warped_image2, x_min, y_min = warp(image2_rgb, H)

#     # Blend the images into a panorama
#     panorama = blend_images(image1_rgb, warped_image2, x_min, y_min)
#     return panorama

def normalize_points(points):
    mean_x, mean_y = np.mean(points, axis=0)

    shifted_points = points - np.array([mean_x, mean_y])

    distances = np.sqrt(np.sum(shifted_points ** 2, axis=1))
    mean_distance = np.mean(distances)

    scale = np.sqrt(2) / mean_distance

    normalized_points = shifted_points * scale

    T = np.array([
        [scale, 0, -scale * mean_x],
        [0, scale, -scale * mean_y],
        [0, 0, 1]
    ])

    return normalized_points, T


def left_to_right_stitching(image_paths):
    images_rgb = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]

    panorama = images_rgb[0]

    for i in range(1, len(images_rgb)):
        print(f"Stitching image {i} onto the panorama")
        panorama = stitch_two_images(panorama, images_rgb[i])
        plt.imshow(panorama)
        plt.axis('off')
        plt.title(f'Mosaic after adding image {i}')
        plt.show()
        mosaic_filename = f'left_to_right_mosaic_{i}.jpg'
        cv2.imwrite(mosaic_filename, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
        print(f"Saved {mosaic_filename}")


    return panorama

def middle_out_stitching(image_paths):
    images_rgb = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]
    num_images = len(images_rgb)


    middle_index = num_images // 2

    panorama = images_rgb[middle_index]

    left_index = middle_index - 1
    right_index = middle_index + 1

    mosaic_counter = 1

    while left_index >= 0 or right_index < num_images:


        if left_index >= 0:
            print(f"Stitching image at index {left_index} onto the left side of the panorama")
            panorama = stitch_two_images(panorama, images_rgb[left_index])
            left_index -= 1

        if right_index < num_images:
            print(f"Stitching image at index {right_index} onto the right side of the panorama")
            panorama = stitch_two_images(panorama, images_rgb[right_index])
            right_index += 1
            
            
        mosaic_filename = f'middle_out_mosaic_{mosaic_counter}.jpg'
        cv2.imwrite(mosaic_filename, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
        print(f"Saved {mosaic_filename}")
        mosaic_counter += 1


    return panorama

def first_out_then_middle_stitching(image_paths):
    images_rgb = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]
    middle_index = len(images_rgb) // 2

    mosaic_counter = 1

    mosaic_left = images_rgb[middle_index - 1]
    for i in range(middle_index - 2, -1, -1):
        print(f"Stitching image {i} onto the left mosaic")
        mosaic_left = stitch_two_images(mosaic_left, images_rgb[i])
        plt.imshow(mosaic_left)
        plt.axis('off')
        plt.title(f'Left mosaic after adding image {i}')
        plt.show()

        mosaic_filename = f'first_out_then_middle_mosaic_left_{mosaic_counter}.jpg'

        cv2.imwrite(mosaic_filename, cv2.cvtColor(mosaic_left, cv2.COLOR_RGB2BGR))
        print(f"Saved {mosaic_filename}")
        mosaic_counter += 1

    mosaic_counter = 1
    mosaic_right = images_rgb[middle_index + 1]
    for i in range(middle_index + 2, len(images_rgb)):
        print(f"Stitching image {i} onto the right mosaic")
        mosaic_right = stitch_two_images(mosaic_right, images_rgb[i])
        plt.imshow(mosaic_right)
        plt.axis('off')
        plt.title(f'Right mosaic after adding image {i}')
        plt.show()

        mosaic_filename = f'first_out_then_middle_mosaic_right_{mosaic_counter}.jpg'
        cv2.imwrite(mosaic_filename, cv2.cvtColor(mosaic_right, cv2.COLOR_RGB2BGR))
        print(f"Saved {mosaic_filename}")
        mosaic_counter += 1

    print("Stitching left mosaic with middle image")
    panorama = stitch_two_images(mosaic_left, images_rgb[middle_index])
    print("Stitching right mosaic onto the panorama")
    panorama = stitch_two_images(panorama, mosaic_right)

    plt.imshow(panorama)
    plt.axis('off')
    plt.title('Final Panorama')
    plt.show()

    mosaic_filename = f'first_out_then_middle_mosaic_final.jpg'
    cv2.imwrite(mosaic_filename, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    print(f"Saved {mosaic_filename}")

    return panorama

def main():
    paris_images = ['paris_a.jpg', 'paris_b.jpg', 'paris_c.jpg']
    print("Task 1: Stitching Paris images")

    paris_panorama = middle_out_stitching(paris_images)

    cv2.imwrite('paris_panorama.jpg', cv2.cvtColor(paris_panorama, cv2.COLOR_RGB2BGR))



    images = ['left_2.jpg', 'left_1.jpg', 'middle.jpg', 'right_1.jpg', 'right_2.jpg']

    # print("\nMethod 1: Left-to-Right")
    # left_to_right_panorama = left_to_right_stitching(images)
    # cv2.imwrite('left_to_right_panorama.jpg', cv2.cvtColor(left_to_right_panorama, cv2.COLOR_RGB2BGR))

    # print("\nMethod 2: Middle-Out")
    # middle_out_panorama = middle_out_stitching(images)
    # cv2.imwrite('middle_out_panorama.jpg', cv2.cvtColor(middle_out_panorama, cv2.COLOR_RGB2BGR))

    # print("\nMethod 3: First-Out-Then-Middle")
    # first_out_panorama = first_out_then_middle_stitching(images)
    # cv2.imwrite('first_out_then_middle_panorama.jpg', cv2.cvtColor(first_out_panorama, cv2.COLOR_RGB2BGR))
    print("Stitching left mosaic with middle image")

    
if __name__ == '__main__':
    main()
