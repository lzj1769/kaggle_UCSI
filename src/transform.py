import math
import numpy as np
import torch
import cv2


def do_resize_image(image, width, height):
    image = cv2.resize(image, dsize=(width, height))
    return image


def do_resize_mask(mask, width, height):
    mask = cv2.resize(mask, dsize=(width, height))
    mask = (mask > 0.5).astype(np.float32)
    return mask


def do_horizontal_flip(image):
    # flip left-right
    image = cv2.flip(image, 1)
    return image


def do_vertical_flip(image):
    # flip left-right
    image = cv2.flip(image, 0)
    return image


def do_shift_scale_crop(image, mask, x0=0, y0=0, x1=1, y1=1):
    # cv2.BORDER_REFLECT_101
    # cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    image = image[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]

    image = cv2.resize(image, dsize=(width, height))
    mask = cv2.resize(mask, dsize=(width, height))
    mask = (mask > 0.5).astype(np.float32)
    return image, mask


def do_random_shift_scale_crop_pad2(image, mask, limit=0.10):
    H, W = image.shape[:2]

    dy = int(H * limit)
    y0 = np.random.randint(0, dy)
    y1 = H - np.random.randint(0, dy)

    dx = int(W * limit)
    x0 = np.random.randint(0, dx)
    x1 = W - np.random.randint(0, dx)

    # y0, y1, x0, x1
    image, mask = do_shift_scale_crop(image, mask, x0, y0, x1, y1)
    return image, mask


def do_horizontal_shear2(image, mask, dx=0):
    borderMode = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    dx = int(dx * width)

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = np.array([[+dx, 0], [width + dx, 0], [width - dx, height], [-dx, height], ], np.float32)

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=borderMode, borderValue=(
            0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,  # cv2.INTER_LINEAR
                               borderMode=borderMode, borderValue=(
            0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = (mask > 0.5).astype(np.float32)
    return image, mask


def do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=0):
    borderMode = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    sx = scale
    sy = scale
    cc = math.cos(angle / 180 * math.pi) * (sx)
    ss = math.sin(angle / 180 * math.pi) * (sy)
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=borderMode, borderValue=(
            0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,  # cv2.INTER_LINEAR
                               borderMode=borderMode, borderValue=(
            0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = (mask > 0.5).astype(np.float32)
    return image, mask


def img_to_tensor(img):
    tensor = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32))
    return tensor


def mask_to_tensor(mask):
    mask = np.expand_dims(mask, 0).astype(np.float32)
    return torch.from_numpy(mask)
