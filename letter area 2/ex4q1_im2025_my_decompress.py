"""
malek ahmad
324921345

in this question i took the data we have in the compresssed file and made the mask matrix again with the real and imaginary data, then i did inverse fft and got the image 
"""

import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt


try:
    with open(sys.argv[1], 'rb') as f:
        original_shape = np.load(f)
        rows, cols = original_shape[0], original_shape[1]
        fshift_real = np.load(f)
        fshift_imag = np.load(f)
        fshift_filtered = fshift_real + 1j * fshift_imag
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_decompressed = np.fft.ifft2(f_ishift)
        img_decompressed = np.abs(img_decompressed)
        plt.imshow(img_decompressed, cmap='gray')
        plt.title('Decompressed Image')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.imwrite('decompressed_image.jpg', img_decompressed)
except FileNotFoundError:
    print(f"{sys.argv[1]} not found.")
    sys.exit(1)
