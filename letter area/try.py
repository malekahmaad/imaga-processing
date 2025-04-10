import cv2
import numpy as np

# Load the image
image = cv2.imread('baffalo.png',  cv2.IMREAD_GRAYSCALE)
red_image = cv2.imread('baffalo.png')
laplacian = cv2.Laplacian(image, cv2.CV_64F)
ret, th = cv2.threshold(laplacian, 35, 255, cv2.THRESH_BINARY)

# Step 2: Iterate through edge pixels and group them
h, w = th.shape
visited = np.zeros_like(th, dtype=bool)
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def dfs(x, y, group):
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if 0 <= cx < w and 0 <= cy < h and th[cy, cx] > 0 and not visited[cy, cx]:
            visited[cy, cx] = True
            group.append((cx, cy))
            stack.extend([(cx + dx, cy + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]])

groups = []
for y in range(h):
    for x in range(w):
        if th[y, x] > 0 and not visited[y, x]:
            group = []
            dfs(x, y, group)
            if len(group) > 50:  # Filter small noisy edge
                groups.append(group)

# Step 3: Draw bounding boxes around groups
for group in groups:
    xs, ys = zip(*group)
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    print(x1, y1, x2, y2)
    print(len(group))
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Manual Edge Grouping', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
