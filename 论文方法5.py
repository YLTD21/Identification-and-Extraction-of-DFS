import cv2
import numpy as np

def extract_river_centerline(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove white pixels from the first and last rows
    gray[0, :] = 0
    gray[-1, :] = 0

    # Apply median blur to remove noise
    blurred = cv2.medianBlur(gray, 5)

    # Enhance edges using an edge-preserving filter (e.g., bilateral filter)
    enhanced_edges = cv2.bilateralFilter(blurred, 9, 75, 75)

    # Apply adaptive thresholding
    _, binary = cv2.threshold(enhanced_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area (assuming it's the river)
    river_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to reduce the number of points
    epsilon = 0.02 * cv2.arcLength(river_contour, True)
    approx_contour = cv2.approxPolyDP(river_contour, epsilon, True)

    # Create an empty mask to draw the river contour
    mask = np.zeros_like(gray)

    # Draw the river contour on the mask
    cv2.drawContours(mask, [approx_contour], 0, (255), thickness=cv2.FILLED)

    # Remove white pixels from the first and last rows in the mask
    mask[0, :] = 0
    mask[-1, :] = 0

    # Apply morphological operations to smooth the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the skeleton of the mask
    skeleton = cv2.ximgproc.thinning(mask)

    # Optionally, remove short branches from the skeleton
    pruned_skeleton = prune_skeleton(skeleton)

    # Process the edges of the skeleton
    processed_skeleton = process_edges(pruned_skeleton)

    return processed_skeleton

def prune_skeleton(skeleton, min_branch_length=40):
    # Convert the skeleton to a binary image
    _, binary_skeleton = cv2.threshold(skeleton, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary skeleton image
    contours, _ = cv2.findContours(binary_skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prune short branches from the skeleton
    pruned_skeleton = np.zeros_like(skeleton)
    for contour in contours:
        if cv2.arcLength(contour, closed=False) > min_branch_length:
            cv2.drawContours(pruned_skeleton, [contour], 0, (255), thickness=cv2.FILLED)

    return pruned_skeleton

def process_edges(skeleton):
    # Find the indices of white pixels in the first and last rows
    first_row_indices = np.where(skeleton[0, :] == 255)[0]
    last_row_indices = np.where(skeleton[-1, :] == 255)[0]

    # If there are multiple white pixels in the first row, keep only the middle one
    if len(first_row_indices) > 1:
        middle_index = len(first_row_indices) // 2
        skeleton[0, :] = 0
        skeleton[0, first_row_indices[middle_index]] = 255

    # If there are multiple white pixels in the last row, keep only the middle one
    if len(last_row_indices) > 1:
        middle_index = len(last_row_indices) // 2
        skeleton[-1, :] = 0
        skeleton[-1, last_row_indices[middle_index]] = 255
        # Process left edge
    left_edge_indices = np.where(skeleton[:, 0] == 255)[0]
    if len(left_edge_indices) > 1:
        middle_index = len(left_edge_indices) // 2
        skeleton[:, 0] = 0
        skeleton[left_edge_indices[middle_index], 0] = 255

    # Process right edge
    right_edge_indices = np.where(skeleton[:, -1] == 255)[0]
    if len(right_edge_indices) > 1:
        middle_index = len(right_edge_indices) // 2
        skeleton[:, -1] = 0
        skeleton[right_edge_indices[middle_index], -1] = 255
    return skeleton

# Example usage:
image_path = r"D:\yjs\826\139.png"
# image_path = r"D:\yjs\826\239.png"
# image_path = r"D:\image.png_20231019090731\134.png"
image = cv2.imread(image_path)
center_line = extract_river_centerline(image)

# Display the original image and the extracted centerline
cv2.imshow("Original Image", image)
cv2.imshow("River Centerline", center_line)
cv2.waitKey(0)
cv2.destroyAllWindows()
