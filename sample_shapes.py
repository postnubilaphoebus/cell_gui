import numpy as np
from scipy.spatial import ConvexHull
import cvxpy

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import gaussian_filter
from detect_ellipsoid_coll import algebraic_separation_condition
from tqdm import tqdm
from scipy.stats import skewnorm
import random
import os
import re
from scipy.ndimage import distance_transform_edt
from scipy.optimize import brentq
from sklearn.decomposition import PCA
from numba import njit
from scipy.ndimage import find_objects
from scipy.ndimage import gaussian_filter
import skimage

def random_rotate_batch(labels):
    batch_size, depth, height, width = labels.shape
    angles = np.random.randint(0, 4, batch_size) * 90  # Multiplying by 90 to get 0, 90, 180, 270
    axes = np.random.randint(0, 3, batch_size)
    rotated_labels = []
    for i in range(batch_size):
        label = labels[i]
        if axes[i] == 0:
            rotated_label = np.rot90(label, k=angles[i] // 90, axes=(1, 2))
        elif axes[i] == 1:
            rotated_label = np.rot90(label, k=angles[i] // 90, axes=(0, 2))
        elif axes[i] == 2:
            rotated_label = np.rot90(label, k=angles[i] // 90, axes=(0, 1))
        rotated_labels.append(rotated_label)
    return np.stack(rotated_labels)

def assign_values_euclidian_3ways_pca(coords, low_value, high_value, cell_decay_base):
    pca = PCA(n_components=3)
    pca.fit(coords.T)
    std_devs = np.sqrt(pca.explained_variance_)
    transformed_coords = pca.transform(coords.T).T
    zero_std_dev_indices = std_devs == 0
    if np.any(zero_std_dev_indices):
        squared_distances = np.zeros(transformed_coords.shape[1])
        for i in range(3):
            if not zero_std_dev_indices[i]:
                squared_distances += (transformed_coords[i, :] / std_devs[i])**2
    else:
        squared_distances = (transformed_coords[0, :] / std_devs[0])**2 + \
                            (transformed_coords[1, :] / std_devs[1])**2 + \
                            (transformed_coords[2, :] / std_devs[2])**2
    sorted_squared_distances = np.sort(squared_distances)
    point_val = sorted_squared_distances[30] 
    point_val2 = sorted_squared_distances[31]
    inbetween_value = (point_val + point_val2) / 2
    max_value = np.max(squared_distances)
    min_value = np.min(squared_distances)
    left_hand_side = (3.0 - low_value) / (high_value - low_value) # 2.0
    def fun(x):
        return (x**(-inbetween_value) - x**(-max_value)) / (x**(-min_value) - x**(-max_value)) - left_hand_side

    a = 1e-3  # lower bound of the interval
    b = 1e5   # upper bound of the interval
    solved_base = brentq(fun, a, b)
    gaussian_values = solved_base ** (-squared_distances)
    min_gaussian = np.min(gaussian_values)
    max_gaussian = np.max(gaussian_values)
    values = low_value + (high_value - low_value) * (gaussian_values - min_gaussian) / (max_gaussian - min_gaussian)
    return values

def transform_labels(labels, min_cell_value, max_cell_value, distance_transform_constant, cell_decay_base, foreground_minimum, background_maximum):
    com_labels = []
    percentage_above_list = []
    for label_image in labels:
        max_index = np.max(label_image.ravel())
        new_label = np.zeros_like(label_image).astype(float)
        ignore_index_array = label_image.copy()
        for i in range(1, max_index+1):
            indices = np.stack(np.where(label_image == i))
            if indices.size > 40: # more than 10 points
                x = indices[0, :]
                y = indices[1, :]
                z = indices[2, :]
                try:
                    scaled_fitted_data = assign_values_euclidian_3ways_pca(indices, foreground_minimum, max_cell_value, cell_decay_base)
                    new_label[x, y, z] = scaled_fitted_data
                except Exception as e: 
                    print(e)
                    ignore_index_array[x, y, z] = -100
            else:
                x = indices[0, :]
                y = indices[1, :]
                z = indices[2, :]
                ignore_index_array[x, y, z] = -100
        foreground_background_inversed = (ignore_index_array == 0).astype(int)
        distance_to_nearest_one = distance_transform_edt(foreground_background_inversed)
        c = distance_transform_constant
        a = c + 1
        negative_inverse_square_distance = (a / (c + distance_to_nearest_one))
        negative_inverse_square_distance = (negative_inverse_square_distance - np.min(negative_inverse_square_distance.ravel())) / (np.max(negative_inverse_square_distance.ravel()) - np.min(negative_inverse_square_distance.ravel()))
        negative_inverse_square_distance = negative_inverse_square_distance * (min_cell_value)
        negative_inverse_square_distance = np.clip(negative_inverse_square_distance, min_cell_value, background_maximum)
        new_label[(ignore_index_array==0) & (ignore_index_array != -100)] = negative_inverse_square_distance[(ignore_index_array==0) & (ignore_index_array != -100)]

        new_label[ignore_index_array == -100] = -100
        com_labels.append(new_label)

    #print("Percentage above 2:", np.mean(percentage_above_list))

    return com_labels

def load_training_images_and_labels(training_path, num_images, image_format=".tif", label_format=".npy"):
    """
    Load training images and corresponding labels from a directory.

    Args:
        training_path (str): Path to the directory containing the training files.
        num_images (int): Number of images to load.
        image_format (str): File format of the images (e.g., '.tif', '.png').
        label_format (str): File format of the labels (e.g., '.npy', '.tif').

    Returns:
        images (list): List of loaded image arrays.
        labels (list): List of corresponding label arrays.
    """
    print("Loading train images and labels...")
    images = []
    labels = []

    # List and sort files in the training path
    files = os.listdir(training_path)
    image_files = sorted([f for f in files 
                          if f.endswith(image_format) 
                          and "label" not in f.lower() 
                          and "mask" not in f.lower()
                          and re.search(r'\d+', f)])
    label_files = sorted([f for f in files 
                          if f.endswith(label_format) 
                          and ("label" in f.lower() or "mask" in f.lower())
                          and re.search(r'\d+', f)])

    for image_file, label_file in zip(image_files[:num_images], label_files[:num_images]):
        # Match numeric IDs in filenames
        image_num = re.search(r'\d+', image_file).group()
        label_num = re.search(r'\d+', label_file).group()

        if image_num == label_num:  # Ensure matching IDs
            imagepath = os.path.join(training_path, image_file)
            labelpath = os.path.join(training_path, label_file)

            # Load the image and label
            image = skimage.io.imread(imagepath)
            if label_format == ".tif":
                label = skimage.io.imread(labelpath).astype(int)
            else:
                label = np.load(labelpath).astype(int)  # Convert labels to integers

            images.append(image)
            labels.append(label)

    return images, labels

class Ellipsoid:
    def __init__(self, position, semi_axes, velocity, orientation):
        self.position = np.array(position)
        self.semi_axes = np.array(semi_axes)
        self.velocity = np.array(velocity)
        self.orientation = np.array(orientation)

def detect_ellipsoid_collision(e1, e2):
    dist = max(np.abs(e1.position - e2.position))
    return dist < (e1.semi_axes.max() + e2.semi_axes.max())

@njit
def detect_ellipsoid_collision_numba(e1_pos, e2_pos, e1_semi_axes, e2_semi_axes):
    dist = np.linalg.norm(e1_pos - e2_pos)
    return dist < (e1_semi_axes.max() + e2_semi_axes.max())

# @njit
# def detect_ellipsoid_collision_numba(e1_pos, e2_pos, e1_semi_axes, e2_semi_axes):
#     dist_squared = np.sum((e1_pos - e2_pos) ** 2)  # Squared distance
#     max_radius_sum = e1_semi_axes.max() + e2_semi_axes.max()
#     return dist_squared < max_radius_sum ** 2

# def resolve_ellipsoid_collision(e1, e2):
#     collision_vector = e2.position - e1.position
#     collision_distance = np.linalg.norm(collision_vector)
    
#     if collision_distance == 0:
#         collision_vector = np.random.uniform(-0.01, 0.01, size=3)
#     else:
#         collision_vector /= collision_distance

#     e1.velocity -= collision_vector
#     e2.velocity += collision_vector

def resolve_ellipsoid_collision(e1, e2, overlap_correction_factor=0.1, damping_factor=0.9, always_resolve=False):
    collision_vector = e2.position - e1.position
    collision_distance = np.linalg.norm(collision_vector)
    min_distance = np.sum(e1.semi_axes) + np.sum(e2.semi_axes)

    if collision_distance == 0:
        collision_vector = np.random.uniform(-0.01, 0.01, size=3)
    else:
        collision_vector /= collision_distance

    overlap = min_distance - collision_distance
    if overlap > 0:
        correction = overlap_correction_factor * overlap * collision_vector
        e1.position -= correction * 0.5
        e2.position += correction * 0.5

    relative_velocity = e1.velocity - e2.velocity
    velocity_along_collision = np.dot(relative_velocity, collision_vector)

    if velocity_along_collision < 0 or always_resolve:  # Only resolve if they are moving towards each other
        impulse = (1 + damping_factor) * velocity_along_collision * collision_vector
        e1.velocity -= impulse * 0.5
        e2.velocity += impulse * 0.5

#@njit
def resolve_ellipsoid_collision_numba(e1_pos, 
                                      e1_semi_axes, 
                                      e1_velocity, 
                                      e2_pos, 
                                      e2_semi_axes, 
                                      e2_velocity,
                                      overlap_correction_factor=0.1, 
                                      damping_factor=0.9, 
                                      always_resolve=False):
    collision_vector = e2_pos- e1_pos
    collision_distance = np.linalg.norm(collision_vector)
    min_distance = np.sum(e1_semi_axes) + np.sum(e2_semi_axes)

    if collision_distance == 0:
        collision_vector = np.random.uniform(-0.01, 0.01, size=3)
    else:
        collision_vector /= collision_distance

    overlap = min_distance - collision_distance
    if overlap > 0:
        correction = overlap_correction_factor * overlap * collision_vector
        e1_pos -= correction * 0.5
        e2_pos+= correction * 0.5

    relative_velocity = e1_velocity- e2_velocity
    velocity_along_collision = np.dot(relative_velocity, collision_vector)

    if velocity_along_collision < 0 or always_resolve:  # Only resolve if they are moving towards each other
        impulse = (1 + damping_factor) * velocity_along_collision * collision_vector
        e1_velocity -= impulse * 0.5
        e2_velocity += impulse * 0.5

    return e1_pos, e2_pos, e1_velocity, e2_velocity

@njit
def resolve_ellipsoid_collision_optimized(e1_pos, 
                                          e1_semi_axes, 
                                          e1_velocity, 
                                          e2_pos, 
                                          e2_semi_axes, 
                                          e2_velocity,
                                          overlap_correction_factor=0.1, 
                                          damping_factor=0.9, 
                                          always_resolve=False):
    collision_vector = e2_pos - e1_pos
    collision_distance_squared = np.sum(collision_vector**2)
    min_distance = np.sum(e1_semi_axes) + np.sum(e2_semi_axes)
    min_distance_squared = min_distance**2

    # Early exit if no collision and not always resolving
    if collision_distance_squared > min_distance_squared and not always_resolve:
        return e1_pos, e2_pos, e1_velocity, e2_velocity

    # Avoid division if distance is zero
    if collision_distance_squared == 0:
        collision_vector = np.array([1e-3, 0, 0])
        collision_distance = 1e-3
    else:
        collision_distance = np.sqrt(collision_distance_squared)
        collision_vector /= collision_distance

    # Overlap correction
    overlap = min_distance - collision_distance
    if overlap > 0:
        correction = overlap_correction_factor * overlap * collision_vector
        e1_pos -= correction * 0.5
        e2_pos += correction * 0.5

    # Velocity correction
    relative_velocity = e1_velocity - e2_velocity
    velocity_along_collision = np.dot(relative_velocity, collision_vector)

    if velocity_along_collision < 0 or always_resolve:
        impulse = (1 + damping_factor) * velocity_along_collision * collision_vector
        e1_velocity -= impulse * 0.5
        e2_velocity += impulse * 0.5

    return e1_pos, e2_pos, e1_velocity, e2_velocity


def generate_random_orientation(axis, angle):
    if axis == 0:
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    else:
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

    return rotation_matrix

@njit
def apply_center_pull(ellipsoid_pos, center, strength=0.01): # 0.01
    direction_to_center = center - ellipsoid_pos
    direction_to_center /= np.linalg.norm(direction_to_center)
    return direction_to_center * strength
    ellipsoid.velocity += direction_to_center * strength

# @njit
# def update_ellipsoid_semi_axes(ellipsoids_sa, growth_rate, max_semi_axes):
#     random_growths = np.random.uniform(-0.7, 0.7, size=(len(ellipsoids_sa), 3))
#     growths = growth_rate * (1 + random_growths)
#     semi_axes = []
#     for growth, sa in zip(growths, ellipsoids_sa):
#         semi_axes.append(np.minimum(sa + growth, max_semi_axes))
#     return semi_axes
@njit
def update_ellipsoid_semi_axes(ellipsoids_sa, growth_rate, max_semi_axes, random_growths):
    growths = growth_rate * (1 + random_growths)
    semi_axes = np.empty_like(ellipsoids_sa)
    for i in range(len(ellipsoids_sa)):
        for j in range(3):
            semi_axes[i, j] = min(ellipsoids_sa[i, j] + growths[i, j], max_semi_axes[j])
    return semi_axes


# def vectorized_gross_collision(positions, semi_axes, tri_indices):
#     pairwise_distances = np.sqrt(np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1))
#     radii_sums = semi_axes.max(axis=1)[:, None] + semi_axes.max(axis=1)[None, :]
#     collision_mask = pairwise_distances < radii_sums
#     colliding_pairs = np.array([[i, j] for i, j in zip(tri_indices[0], tri_indices[1]) if collision_mask[i, j]])
#     return colliding_pairs

@njit
def vectorized_gross_collision(positions, semi_axes, tri_indices):
    # Compute pairwise squared distances for the specific indices in tri_indices
    pairwise_distances_squared = np.sum((positions[tri_indices[0]] - positions[tri_indices[1]]) ** 2, axis=-1)
    max_radii_0 = np.zeros(len(semi_axes))

    for i in range(len(semi_axes)):
        max_radii_0[i] = np.max(semi_axes[i])  # Max semi-axis for ellipsoid i

    # Now compute the sum of the max semi-axes for each pair
    radii_sums = max_radii_0[tri_indices[0]] + max_radii_0[tri_indices[1]]
    collision_mask = pairwise_distances_squared < radii_sums ** 2
    colliding_pairs = np.column_stack((tri_indices[0][collision_mask], tri_indices[1][collision_mask]))
    return colliding_pairs


from scipy.spatial import KDTree

# sparse_distance_matrix

def lubachevsky_stillinger_ellipsoids(random_seed, growth_rate, max_steps, max_semi_axes, max_orientation_angle):    
    ellipsoids = []
    max_steps = 650
    pause_growth_interval = 2
    volume_limit = 63
    center = np.array([volume_limit / 2] * 3)
    extra_row = np.array([0, 0, 0, 1]).reshape(1, -1)   
    from scipy.stats.qmc import PoissonDisk
    dimensions = 3            # 3D space
    #radius = 0.14         # Minimum distance between points
    radius = np.random.uniform(0.135, 0.145)
    sampler = PoissonDisk(d=dimensions, radius=radius, seed = random_seed)

    # Generate samples
    samples = sampler.fill_space()

    # Convert samples to a NumPy array for processing
    samples = np.array(samples)

    


    samples = samples * 63

    kdtree_samples = KDTree(samples)

    sparse_matrix = kdtree_samples.sparse_distance_matrix(kdtree_samples, max_distance = 13)
    sparse_matrix = sparse_matrix.toarray()
    # Create a boolean mask for positive entries
    positive_mask = sparse_matrix > 0

    # Extract the upper triangle indices (excluding the diagonal)
    row_indices, col_indices = np.where(np.triu(positive_mask, k=1))
    tri_indices = tuple((row_indices, col_indices))
    # unique_indices = np.unique(tri_indices)
    # import pdb; pdb.set_trace()
    #print("shape", samples.shape)
    #import pdb; pdb.set_trace()
    positions = samples
    num_ellipsoids = len(positions)
    mean_growth_rate = 0.01

    #tri_indices = np.triu_indices(num_ellipsoids, k=1)

    for position in positions:
        velocity = np.random.uniform(-1, 1, size=3)
        direction_to_center = center - position
        norm = np.linalg.norm(direction_to_center)
        if norm > 0:  # Avoid division by zero
            velocity = direction_to_center / norm
        else:
            velocity = np.random.uniform(-1, 1, size=3)
        rot_axis = np.random.choice([0, 1])
        max_angle_radians = np.radians(max_orientation_angle)
        angle = np.random.uniform(0, max_angle_radians)
        orientation = generate_random_orientation(rot_axis, angle)
        initial_semi_axes = np.random.rand(3) * 1.8 + 1.0
        ellipsoids.append(Ellipsoid(position, initial_semi_axes, velocity, orientation))

    import time

    for step in range(max_steps):
        t0 = time.time()
        if step % pause_growth_interval == 0:
            random_growths = np.random.uniform(-0.7, 0.7, size=(len(ellipsoids), 3))
            current_semi_axes = np.array([ellipsoid.semi_axes for ellipsoid in ellipsoids])
            semi_axes = update_ellipsoid_semi_axes(current_semi_axes, growth_rate, max_semi_axes, random_growths)
            for axis, ellipsoid in zip(semi_axes, ellipsoids):
                ellipsoid.semi_axes = axis

            # current_semi_axes = np.array([ellipsoid.semi_axes for ellipsoid in ellipsoids])
            # semi_axes = update_ellipsoid_semi_axes(current_semi_axes, growth_rate, max_semi_axes)
            # for axis, ellipsoid in zip(semi_axes, ellipsoids):
            #     ellipsoid.semi_axes = axis

        positions = np.array([ellipsoid.position for ellipsoid in ellipsoids])
        semi_axes = np.array([ellipsoid.semi_axes for ellipsoid in ellipsoids])
        t1 = time.time()
        colliding_pairs = vectorized_gross_collision(positions, semi_axes, tri_indices)
        t2 = time.time()
        if colliding_pairs.size > 0:
            for pair in colliding_pairs:
                expanded_pos_i = ellipsoids[pair[0]].position.reshape(1, -1)
                expanded_pos_j = ellipsoids[pair[1]].position.reshape(1, -1)
                e1_pos = ellipsoids[pair[0]].position
                e1_semi_axes = ellipsoids[pair[0]].semi_axes
                e1_velocity = ellipsoids[pair[0]].velocity
                e2_pos = ellipsoids[pair[1]].position
                e2_semi_axes = ellipsoids[pair[1]].semi_axes
                e2_velocity = ellipsoids[pair[1]].velocity
                if algebraic_separation_condition(ellipsoids[pair[0]].semi_axes, 
                                                ellipsoids[pair[1]].semi_axes, 
                                                expanded_pos_i, 
                                                expanded_pos_j, 
                                                ellipsoids[pair[0]].orientation, 
                                                ellipsoids[pair[1]].orientation,
                                                extra_row):
                    e1_pos, e2_pos, e1_velocity, e2_velocity = resolve_ellipsoid_collision_numba(e1_pos, 
                                                                                                e1_semi_axes, 
                                                                                                e1_velocity, 
                                                                                                e2_pos, 
                                                                                                e2_semi_axes, 
                                                                                                e2_velocity,
                                                                                                overlap_correction_factor=0.1,
                                                                                                damping_factor=0.9,
                                                                                                always_resolve=False)
                    ellipsoids[pair[0]].position = e1_pos
                    ellipsoids[pair[1]].position = e2_pos
                    ellipsoids[pair[0]].velocity = e1_velocity
                    ellipsoids[pair[1]].velocity = e2_velocity

                    # if step >50:
                    #     t3 = time.time()
                    #     total_time = t3 - t0
                    #     first_time_fraction = (t1 - t0) / total_time
                    #     second_time_fraction = (t2 - t1) / total_time
                    #     third_time_fraction = (t3 - t2) / total_time
                    #     print(f"Total time: {total_time}")
                    #     print(f"First time fraction: {first_time_fraction}")
                    #     print(f"Second time fraction: {second_time_fraction}")
                    #     print(f"Third time fraction: {third_time_fraction}")
                    #     print("second time total: ", t2 - t1)
                    #     #import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()



        # #t1 = time.time()
        # for i in range(num_ellipsoids):
        #     for j in range(i + 1, num_ellipsoids):
        #         e1_pos = ellipsoids[i].position
        #         e2_pos = ellipsoids[j].position
        #         e1_semi_axes = ellipsoids[i].semi_axes
        #         e2_semi_axes = ellipsoids[j].semi_axes
        #         e1_velocity = ellipsoids[i].velocity
        #         e2_velocity = ellipsoids[j].velocity
        #         if detect_ellipsoid_collision_numba(e1_pos, e2_pos, e1_semi_axes, e2_semi_axes):
        #             #t2 = time.time()
        #             expanded_pos_i = ellipsoids[i].position.reshape(1, -1)
        #             expanded_pos_j = ellipsoids[j].position.reshape(1, -1)
        #             if algebraic_separation_condition(ellipsoids[i].semi_axes, 
        #                                               ellipsoids[j].semi_axes, 
        #                                               expanded_pos_i, 
        #                                               expanded_pos_j, 
        #                                               ellipsoids[i].orientation, 
        #                                               ellipsoids[j].orientation,
        #                                               extra_row):
        #                 e1_pos, e2_pos, e1_velocity, e2_velocity = resolve_ellipsoid_collision_numba(e1_pos, 
        #                                                                                                  e1_semi_axes, 
        #                                                                                                  e1_velocity, 
        #                                                                                                  e2_pos, 
        #                                                                                                  e2_semi_axes, 
        #                                                                                                  e2_velocity,
        #                                                                                                  overlap_correction_factor=0.1,
        #                                                                                                  damping_factor=0.9,
        #                                                                                                  always_resolve=False)
        #                 ellipsoids[i].position = e1_pos
        #                 ellipsoids[j].position = e2_pos
        #                 ellipsoids[i].velocity = e1_velocity
        #                 ellipsoids[j].velocity = e2_velocity

        #t2 = time.time()

        for ellipsoid in ellipsoids:
            ellipsoid.velocity += apply_center_pull(ellipsoid.position, center)
            #import pdb; pdb.set_trace()
            ellipsoid.position += ellipsoid.velocity * mean_growth_rate#np.abs(np.mean(growth_rate))
            # if np.any((growth_rate!= 0.01)):
            #     import pdb; pdb.set_trace()
            for k in range(3):
                if ellipsoid.position[k] < -3 or ellipsoid.position[k] > 67:
                    # Restrict position to bounds
                    ellipsoid.position[k] = max(-3, min(ellipsoid.position[k], 67))
                    
                    # Update velocity to point towards the center
                    direction_to_center = center - ellipsoid.position
                    norm = np.linalg.norm(direction_to_center)
                    if norm > 0:  # Avoid division by zero
                        ellipsoid.velocity = direction_to_center / norm
            # for k in range(3):
            #     if ellipsoid.position[k] < -3 or ellipsoid.position[k] > 67:
            #         direction_to_center = center - ellipsoid.position
            #         ellipsoid.velocity = direction_to_center / np.linalg.norm(direction_to_center)
            #         if ellipsoid.position[k] < -3:
            #             ellipsoid.position[k] = -3
            #         if ellipsoid.position[k] > 67:
            #             ellipsoid.position[k] = 67

        #t3 = time.time()
        # total_time = t3 - t0
        # first_time_fraction = (t1 - t0) / total_time
        # second_time_fraction = (t2 - t1) / total_time
        # third_time_fraction = (t3 - t2) / total_time

        # print(f"First time fraction: {first_time_fraction}")
        # print(f"Second time fraction: {second_time_fraction}")
        # print(f"Third time fraction: {third_time_fraction}")

        # import pdb; pdb.set_trace()
    #print("resolving final collisions...")
    new_growth_rate = np.array([0.01, 0.01, 0.01])
    collision_detected = True
    oob = True
    while collision_detected or oob:
        collision_detected = False
        oob = False
        for i in range(num_ellipsoids):
            for j in range(i + 1, num_ellipsoids):
                e1_pos = ellipsoids[i].position
                e2_pos = ellipsoids[j].position
                e1_semi_axes = ellipsoids[i].semi_axes
                e2_semi_axes = ellipsoids[j].semi_axes
                e1_velocity = ellipsoids[i].velocity
                e2_velocity = ellipsoids[j].velocity
                if detect_ellipsoid_collision_numba(e1_pos, e2_pos, e1_semi_axes, e2_semi_axes):
                    expanded_pos_i = ellipsoids[i].position.reshape(1, -1)
                    expanded_pos_j = ellipsoids[j].position.reshape(1, -1)
                    
                    if algebraic_separation_condition(ellipsoids[i].semi_axes, 
                                                      ellipsoids[j].semi_axes, 
                                                      expanded_pos_i, 
                                                      expanded_pos_j, 
                                                      ellipsoids[i].orientation, 
                                                      ellipsoids[j].orientation,
                                                      extra_row): 
                        # resolve_ellipsoid_collision(ellipsoids[i], ellipsoids[j], always_resolve=True)
                        e1_pos, e2_pos, e1_velocity, e2_velocity = resolve_ellipsoid_collision_numba(e1_pos, 
                                                            e1_semi_axes, 
                                                            e1_velocity, 
                                                            e2_pos, 
                                                            e2_semi_axes, 
                                                            e2_velocity,
                                                            overlap_correction_factor=0.1,
                                                            damping_factor=0.9,
                                                            always_resolve=True)
                        ellipsoids[i].position = e1_pos
                        ellipsoids[j].position = e2_pos
                        ellipsoids[i].velocity = e1_velocity
                        ellipsoids[j].velocity = e2_velocity
                        collision_detected = True
        for ellipsoid in ellipsoids:
            ellipsoid.position += ellipsoid.velocity * mean_growth_rate
            for k in range(3):
                if ellipsoid.position[k] < -3 or ellipsoid.position[k] > 67:
                    ellipsoid.position[k] = max(-3, min(ellipsoid.position[k], 67))
                    direction_to_center = center - ellipsoid.position
                    norm = np.linalg.norm(direction_to_center)
                    if norm > 0:  
                        ellipsoid.velocity = direction_to_center / norm
                    oob = True
            # for k in range(3):
            #     if ellipsoid.position[k] < -3 or ellipsoid.position[k] > 67:
            #         direction_to_center = center - ellipsoid.position
            #         ellipsoid.velocity = direction_to_center / np.linalg.norm(direction_to_center)
            #         if ellipsoid.position[k] < -3:
            #             ellipsoid.position[k] = -3
            #         if ellipsoid.position[k] > 67:
            #             ellipsoid.position[k] = 67
                    #ellipsoid.position[k] = -3
                    # oob = True
                    

    return ellipsoids

@njit
def is_point_inside_ellipsoid(points, e_pos, e_orient, e_semi_axes):
    translated_points = points - e_pos
    rotated_points = translated_points @ e_orient.T
    normalized_points = rotated_points / e_semi_axes
    return np.sum(normalized_points**2, axis=1) <= 1

def pack_labels(volume):
    unique_labels = np.unique(volume)
    unique_labels = unique_labels[unique_labels != 0]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}
    packed_volume = np.copy(volume)
    for old_label, new_label in label_mapping.items():
        packed_volume[volume == old_label] = new_label

    return packed_volume

def random_rotate_3d_cube(volume):
    # Randomly select one of the three axes pairs to rotate around
    axes = [(0, 1), (1, 2), (0, 2)]
    selected_axes = axes[np.random.randint(0, len(axes))]
    
    # Randomly select an angle (90, 180, 270 degrees)
    k = np.random.choice([1, 2, 3])
    
    # Rotate the volume
    rotated_volume = np.rot90(volume, k=k, axes=selected_axes)
    
    return rotated_volume

def generate_3d_volume(ellipsoids, volume_size):
    x, y, z = np.indices((volume_size, volume_size, volume_size))
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)  
    volume = np.zeros((volume_size, volume_size, volume_size), dtype=np.uint16)
    label_volume = np.zeros((volume_size, volume_size, volume_size), dtype=np.uint16)

    for idx, ellipsoid in enumerate(ellipsoids, start=1):
        inside = is_point_inside_ellipsoid(points, ellipsoid.position, ellipsoid.orientation, ellipsoid.semi_axes)
        volume.ravel()[inside] = 1
        label_volume.ravel()[inside] = idx

    label_volume = pack_labels(label_volume)
    #print("num cells", max(label_volume.ravel()))
    return volume, label_volume

def compute_volume(semi_axes):
    return 4/3 * np.pi * semi_axes[0] * semi_axes[1] * semi_axes[2]

def compute_density(ellipsoids, volume):
    total_volume = sum(compute_volume(e.semi_axes) for e in ellipsoids)
    return total_volume / volume

def generate_synthetic_volumes(random_seed, max_steps):
    import time

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Parameters
    volume_size = 64
    growth_rate = np.array([0.01, 0.01, 0.01]) # growth_rate = np.array([0.01, 0.01, 0.01])
    #growth_rate = np.array([0.012, 0.012, 0.012])
    max_semi_axes = np.array([12, 12, 12])
    max_orientation_angle = 360  # Maximum angle in degrees from the xy-plane

    ellipsoids = lubachevsky_stillinger_ellipsoids(random_seed, growth_rate, max_steps, max_semi_axes, max_orientation_angle)

    volume, label_volume = generate_3d_volume(ellipsoids, volume_size)
    #label_volume = label_volume.astype(np.uint16)

    slices = find_objects(label_volume)
    label_volume_shape = label_volume.shape
    label_heatmap = np.random.rand(*label_volume_shape)
    label_heatmap = -5.0 + (-3.5 + 5.0) * label_heatmap
    label_heatmap = label_heatmap.astype(np.float64)

    for i, slice_tuple in enumerate(slices):
        if slice_tuple is not None:  # Check if the slice is not None
            ellipsoid_locs = np.argwhere(label_volume[slice_tuple] == (i + 1)) + np.array([s.start for s in slice_tuple])
            if ellipsoid_locs.shape[0] > 40:
                pca_values = assign_values_euclidian_3ways_pca(ellipsoid_locs.T, -2.0, 20.0, 5.0)
                pca_values = pca_values * (1 + (np.random.rand(1) - 0.5) * 0.3)
                label_heatmap[ellipsoid_locs[:, 0], ellipsoid_locs[:, 1], ellipsoid_locs[:, 2]] = pca_values

    label_heatmap = gaussian_filter(label_heatmap, sigma=1.0)
    label_heatmap = (label_heatmap - label_heatmap.min()) / (label_heatmap.max() - label_heatmap.min())
    return label_heatmap, label_volume

def GetHull(points):
    '''Function author: Raluca Sandu'''
    dim = points.shape[1]
    hull = ConvexHull(points)
    A = hull.equations[:,0:dim] # vectors for each facet
    b = hull.equations[:,dim] # distance from origin of each facet
    return A, -b, hull  # Negative moves b to the RHS of the inequality


def inner_ellipsoid_fit(points, verbose=False):
    # ellipsoid_matrix, ellipsoid_centroid = inner_ellipsoid_fit(points)
    '''Function author: Raluca Sandu'''
    """Find the inscribed ellipsoid into a set of points of maximum volume."""
    points = np.array(points)
    dim = points.shape[1]
    A,b,hull = GetHull(points)

    B = cvxpy.Variable((dim,dim), PSD=True)  # Ellipsoid
    d = cvxpy.Variable(dim)                  # Center

    constraints = [cvxpy.norm(B @ A[i],2) + A[i] @ d <= b[i] for i in range(len(A))]
    prob = cvxpy.Problem(cvxpy.Minimize(-cvxpy.log_det(B)), constraints)
    optval = prob.solve(solver=cvxpy.MOSEK)
    if optval==np.inf:
        raise Exception("No solution possible!")
    if verbose:
        print(f"Optimal value: {optval}")
    
    return B.value, d.value

# first step: describe all possible inner ellipsoids
# sample between 250 and 350 ellipsoids
# second step: put each ellipsoid in a cube as large as the largest possible axis length (muktiply all axes by 1.1)
# third step: make the ellipsoids move closer according to some simple rule system (rejection sampling?)




saving_path = "/Users/lauridsstockert/Desktop/ellipsoid_generation/generated_data"
import skimage
volume_size = 64
for i in tqdm(range(0, 100)):
    noisy_blurred_volume, label_volume = generate_synthetic_volumes(random_seed=i, max_steps=650) # max_steps=650
    fpath = os.path.join(saving_path, "image"+str(i) + '.tif')

    tifffile.imwrite(fpath, noisy_blurred_volume)

    np.save(os.path.join(saving_path, "label"+str(i) + '.npy'), label_volume)






    # np.save("noisy"+str(i) + '.npy', label_volume)

    # # Plot slices
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # slices = [noisy_blurred_volume[volume_size // 4, :, :], noisy_blurred_volume[volume_size // 2, :, :], noisy_blurred_volume[3 * volume_size // 4, :, :]]

    # for i, ax in enumerate(axes):
    #     ax.imshow(slices[i], cmap='gray')
