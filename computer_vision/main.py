import numpy as np
import matplotlib.pyplot as plt
import cv2
from edge_detection import compute_fg_edges
from generate_img import generate_image
from gen_reflectance_map import compute_reflectance
from minimisation import update_fg
from depth_reconstruction import integrate_depth

# IMAGE GENERATION
# E, normals_gt, z_gt = generate_image(64, save=True, return_depth=True, s_perturb=False)

# LOADING IMAGES
# Load image from a folder in the working directory
image_path = "coil-20-proc/obj1__18.png"  # Replace with the actual folder and file name
image = cv2.imread(image_path)   # Reads image in BGR format

# Convert to grayscale if needed
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Display the image
cv2.imshow("Input Image", gray_image)
cv2.waitKey(0)  # Waits for a key press
cv2.destroyAllWindows()
E = np.array(gray_image, dtype=np.float64)
# Normalize to [0,1] range for processing if needed
E /= 255.0 

# EDGE DETECTION AND INITIALISATION
f, g, N_vectors, mask = compute_fg_edges(E,save=True)

# SOLVING THE PDE FOR MINIMISATION PROBLEM
lambda_reg = 100.0
max_iterations = 100
tol = 1e-3

f, g = update_fg(f, g, E, mask, lambda_reg,max_iterations,tol)

# DEPTH RECONSTRUCTION
z = integrate_depth(f, g, mask, z_gt=0, visual=True, scale=True)

plt.show()
