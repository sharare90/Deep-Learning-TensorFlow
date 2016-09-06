import numpy as np

import settings


thresholds = {
    0: 0,
    128: 1,
    192: 2,
    254: 3,
}


def convert_label_to_thresholds(element):
    return thresholds[element]


convert_label_to_thresholds = np.vectorize(convert_label_to_thresholds)


def pre_process(image, label):
    label = label.reshape([settings.HEIGHT, settings.WIDTH])
    image = image.reshape(256 * 256, 1)
    image = image - min(image)
    image = image / float(max(image))
    image = image.reshape(256, 256)
    return image, convert_label_to_thresholds(label)


def windowing(image, label, height=settings.HEIGHT, width=settings.WIDTH, window_height=settings.WINDOW_HEIGHT,
              window_width=settings.WINDOW_WIDTH, column_format=False, preprocess=False):
    if preprocess:
        image, label = pre_process(image, label)
    borderless_hieght = height - (window_height - 1)
    borderless_width = width - (window_width - 1)
    imgs = []
    labels = []
    for j in xrange(borderless_hieght):
        for k in xrange(borderless_width):
            img = image[j:window_height + j, k:window_width + k]
            lbl = label[j + window_height / 2, k + window_width / 2]
            if column_format:
                img = img.reshape(img.shape[0] * img.shape[1], )
            imgs.append(img)
            labels.append(lbl)

    return imgs, labels
