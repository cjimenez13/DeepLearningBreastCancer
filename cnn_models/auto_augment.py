import random
import numpy as np
import scipy
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps
import skimage as sk


operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: invert(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Solarize': lambda img, magnitude: solarize(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Color': lambda img, magnitude: color(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: cutout(img, magnitude),
}


def apply_policy(img, policy):
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])

    return img


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix


def shear_x(img, magnitude):
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def shear_y(img, magnitude):
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def translate_x(img, magnitude):
    tform = sk.transform.SimilarityTransform(translation=(magnitude, 0))
    return sk.transform.warp(img, tform.inverse)
    # magnitudes = np.linspace(-150/331, 150/331, 11)

    # transform_matrix = np.array([[1, 0, 0],
    #                              [0, 1, img.shape[1]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
    #                              [0, 0, 1]])
    # transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    # affine_matrix = transform_matrix[:2, :2]
    # offset = transform_matrix[:2, 2]
    # img = np.stack([ndimage.interpolation.affine_transform(
    #                 img[:, :, c],
    #                 affine_matrix,
    #                 offset) for c in range(img.shape[2])], axis=2)
    # return img


def translate_y(img, magnitude):
    tform = sk.transform.SimilarityTransform(translation=(0, magnitude))
    return sk.transform.warp(img, tform.inverse)
    # magnitudes = np.linspace(-150/331, 150/331, 11)

    # transform_matrix = np.array([[1, 0, img.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
    #                              [0, 1, 0],
    #                              [0, 0, 1]])
    # transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    # affine_matrix = transform_matrix[:2, :2]
    # offset = transform_matrix[:2, 2]
    # img = np.stack([ndimage.interpolation.affine_transform(
    #                 img[:, :, c],
    #                 affine_matrix,
    #                 offset) for c in range(img.shape[2])], axis=2)
    # return img


def rotate(img, magnitude):
    return sk.transform.rotate(img, magnitude)
    # magnitudes = np.linspace(-30, 30, 11)

    # theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    # transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                              [np.sin(theta), np.cos(theta), 0],
    #                              [0, 0, 1]])
    # transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    # affine_matrix = transform_matrix[:2, :2]
    # offset = transform_matrix[:2, 2]
    # img = np.stack([ndimage.interpolation.affine_transform(
    #                 img[:, :, c],
    #                 affine_matrix,
    #                 offset) for c in range(img.shape[2])], axis=2)
    # return img


def auto_contrast(img, magnitude):
    
    img = Image.fromarray((img[0] * 255).astype(np.uint8))
    img = ImageOps.autocontrast(img)
    img = np.array(img)
    return img


def invert(img, magnitude):
    
    img = Image.fromarray((img[0] * 255).astype(np.uint8))
    img = ImageOps.invert(img)
    img = np.array(img)
    return img


def equalize(img, magnitude):
    
    img = Image.fromarray((img[0] * 255).astype(np.uint8))
    img = ImageOps.equalize(img)
    img = np.array(img)
    return img


def solarize(img, magnitude):
    magnitudes = np.linspace(0, 256, 11)

    
    img = Image.fromarray((img[0] * 255).astype(np.uint8))
    img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    img = np.array(img)
    return img


def posterize(img, magnitude):
    magnitudes = np.linspace(4, 8, 11)
    
    img = Image.fromarray((img[0] * 255).astype(np.uint8))
    img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))))
    img = np.array(img)
    return img


def contrast(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    
    img = Image.fromarray((img[0] * 255).astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    img = np.array(img)
    return img


def color(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    
    img = Image.fromarray((img[0] * 255).astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    img = np.array(img)
    return img


def brightness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    
    img = Image.fromarray((img[0] * 255).astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    img = np.array(img)
    return img


def sharpness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    
    img = Image.fromarray((img[0] * 255).astype(np.uint8))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    img = np.array(img)
    return img


def cutout(org_img, magnitude=None):
    magnitudes = np.linspace(0, 60/331, 11)

    img = np.copy(org_img)
    mask_val = img.mean()

    if magnitude is None:
        mask_size = 16
    else:
        mask_size = int(round(img.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])))
    top = np.random.randint(0 - mask_size//2, img.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size//2, img.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    img[top:bottom, left:right, :].fill(mask_val)

    return img
