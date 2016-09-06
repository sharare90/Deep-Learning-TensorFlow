import numpy as np
from ibsrfiles.utils import windowing

image = np.ndarray(shape=[25, 25])
label = np.ndarray(shape=[25, 25])
for i in xrange(25):
    for j in xrange(25):
        image[i, j] = 25 * i + j
        label[i, j] = 25 * i + j

images, lables = windowing(image, label, height=25, width=25, window_height=5, window_width=5)

expected_image = np.array([
    [29, 30, 31, 32, 33],
    [54, 55, 56, 57, 58],
    [79, 80, 81, 82, 83],
    [104, 105, 106, 107, 108],
    [129, 130, 131, 132, 133]
])

if (images[25] == expected_image).min():
    print 'Everything is OK.'
else:
    print 'Oops there is a problem.'
