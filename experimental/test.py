import cv2

# Load the image
image = cv2.imread('datasets/hq_object.png', cv2.IMREAD_GRAYSCALE)
max_value = image.max()
min_value = image.min()

print(f"Max value: {max_value}")
print(f"Min value: {min_value}")
