import os, random
import numpy as np
from PIL import Image, ImageEnhance

import jittor as jt
from jittor import dataset


def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if (flip_flag == 1):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return (img, label)


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint((image_width - border), image_width)
    crop_win_height = np.random.randint((image_height - border), image_height)
    random_region = (((image_width - crop_win_width) >> 1), ((image_height - crop_win_height) >> 1), ((image_width + crop_win_width) >> 1), ((image_height + crop_win_height) >> 1))
    return (image.crop(random_region), label.crop(random_region))


def randomRotation(image, label):
    mode = Image.BICUBIC
    if (random.random() > 0.8):
        random_angle = np.random.randint((- 15), 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return (image, label)


def colorEnhance(image):
    bright_intensity = (random.randint(5, 15) / 10.0)
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = (random.randint(5, 15) / 10.0)
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = (random.randint(0, 20) / 10.0)
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = (random.randint(0, 30) / 10.0)
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):

    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    (width, height) = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(((0.0015 * img.shape[0]) * img.shape[1]))
    for i in range(noiseNum):
        randX = random.randint(0, (img.shape[0] - 1))
        randY = random.randint(0, (img.shape[1] - 1))
        if (random.randint(0, 1) == 0):
            img[(randX, randY)] = 0
        else:
            img[(randX, randY)] = 255
    return Image.fromarray(img)


class PolypObjDataset(dataset.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        super().__init__()
        self.trainsize = trainsize
        self.images = [(image_root + f) for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [(gt_root + f) for f in os.listdir(gt_root) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.img_transform = jt.transform.Compose([
            jt.transform.Resize((self.trainsize, self.trainsize)),
            jt.transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            jt.transform.ToTensor()
        ])
        self.gt_transform = jt.transform.Compose([
            jt.transform.Resize((self.trainsize, self.trainsize)),
            jt.transform.ToTensor()
        ])
        self.totensor = jt.transform.ToTensor()
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        (image, gt) = cv_random_flip(image, gt)
        (image, gt) = randomCrop(image, gt)
        (image, gt) = randomRotation(image, gt)
        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        # gt = np.expand_dims(np.array(gt), axis=0)

        return (image, gt)

    def filter_files(self):
        assert ((len(self.images) == len(self.gts)) and (len(self.gts) == len(self.images)))
        images = []
        gts = []
        for (img_path, gt_path) in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if (img.size == gt.size):
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, trainsize):
    dataset = PolypObjDataset(image_root, gt_root, trainsize)
    return dataset


class test_dataset(dataset.Dataset):
    def __init__(self, image_root, gt_root, testsize):
        super().__init__(self)
        self.testsize = testsize
        self.images = [(image_root + f) for f in os.listdir(image_root) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.gts = [(gt_root + f) for f in os.listdir(gt_root) if (f.endswith('.tif') or f.endswith('.png'))]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = jt.transform.Compose([
            jt.transform.Resize((self.testsize, self.testsize)),
            jt.transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            jt.transform.ToTensor()
        ])
        self.gt_transform = jt.transform.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[(- 1)]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = (name.split('.jpg')[0] + '.png')
        self.index += 1
        self.index = (self.index % self.size)
        return (image, gt, name, np.array(image_for_post))

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
