"""
malek ahmad
324921345

in this question i did fft and fft shift and got the fourier transform then i wanted to takke the important data (i want a low pass mask) so i made a mask with white circle in it and 
then multiply it with the fourier transform matrix and i got a mask with imaginary and real numbers so i saved them and saved the image shape
"""

import numpy as np
import cv2
import sys

if len(sys.argv) < 2:
    print("You need to write the image filename.")
    sys.exit(1)

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
rows, cols = img.shape
new_row, new_col = rows // 2, cols // 2
mask = np.zeros((rows, cols), dtype=np.float64)
radius = rows // 3
cv2.circle(mask, (new_col, new_row), radius, 1, thickness=-1)
fshift_filtered = fshift * mask

with open('compressed_image.dat', 'wb') as f:
    np.save(f, np.array([rows, cols], dtype=np.uint32))
    np.save(f, np.real(fshift_filtered))
    np.save(f, np.imag(fshift_filtered))
