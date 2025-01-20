import numpy as np
import skimage
import itertools
from tqdm import tqdm
import time
from skimage.segmentation import watershed
from scipy.ndimage import label, find_objects, distance_transform_cdt
from skimage.feature import peak_local_max
import numpy as np
from constants import extra_padding_width, minimum_cell_size_boundary
import os
from tqdm import tqdm
from skimage import morphology
import tifffile
from skimage.graph import pixel_graph, cut_normalized
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import correlate
from numba import njit, prange
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
import re
import os
from scipy.optimize import brentq
from sklearn.decomposition import PCA
from scipy.ndimage import find_objects


post_processing_centre_threshold = 0.25
postprocessing_background_threshold = 0.07

def generate_3x3x3_volume_without_center(center_point, image_shape):
    x, y, z = center_point
    offsets = np.array([[i, j, k] for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)])
    surrounding_points = offsets + np.array([x, y, z])
    return surrounding_points

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


def raining_watershed(prediction, 
                      predicted_label_path, 
                      inference_filename):
    inference = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    foreground_bool = inference > post_processing_centre_threshold
    #maxima = local_maxima(inference)
    maxima_locs = peak_local_max(inference, min_distance=2, threshold_rel = 0.18, p_norm=2.0)
    maxima = np.zeros_like(inference).astype(int)
    maxima[maxima_locs[:, 0], maxima_locs[:, 1], maxima_locs[:, 2]] = 1
    #import pdb; pdb.set_trace()
    #maxima = maxima * foreground_bool
    labeled_array, num_features = label(maxima)
    distance_map, indices = distance_transform_edt(labeled_array == 0, return_distances=True, return_indices=True)
    nearest_indices = labeled_array[tuple(indices)]
    background_image = (inference > postprocessing_background_threshold).astype(int)
    inference_inverted = 1.0 - inference
    background_image[0, :, :] = 0  
    background_image[-1, :, :] = 0  
    background_image[:, 0, :] = 0 
    background_image[:, -1, :] = 0 
    background_image[:, :, 0] = 0 
    background_image[:, :, -1] = 0  
    points_to_be_assigned = np.argwhere(background_image)
    neighbor_gradient_dict = {tuple(key): None for key in points_to_be_assigned}

    # find flow directions
    coordinates = np.array([(x, y, z) for x in range(3) for y in range(3) for z in range(3)])
    center = np.array([1, 1, 1])
    distances = np.linalg.norm(coordinates - center, axis=1) # euclidian
    distances[13] = 1.0
    #import pdb; pdb.set_trace()
    #distances[distances > 1] = 0.0001
    #import pdb; pdb.set_trace()
    #distances = np.sum(np.abs(coordinates - center), axis=1) # manhattan, works also
    for point in tqdm(points_to_be_assigned):
        neighbors = generate_3x3x3_volume_without_center(point, inference_inverted.shape)
        neighbor_values = inference_inverted[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]]
        point_value = inference_inverted[point[0], point[1], point[2]]
        diffs = point_value - neighbor_values
        diffs = diffs / distances # normalize flow direction by distances
        lowest_diff_loc = np.argmin(diffs)
        lowest_diff = diffs[lowest_diff_loc]
        #print("lowest diff", lowest_diff)
        if lowest_diff < 0:
            #import pdb; pdb.set_trace()
            highest_descent_neighbor = neighbors[lowest_diff_loc]
            neighbor_gradient_dict[tuple(point)] = tuple(highest_descent_neighbor)

    # basin_number = 1

    # for key in neighbor_gradient_dict.keys():
    #     val = neighbor_gradient_dict.get(key)
    #     if val is None:
    #         neighbor_gradient_dict[key] = basin_number
    #         basin_number += 1

    # import pdb; pdb.set_trace()


    

    # let points flow
    wts = np.zeros(inference_inverted.shape).astype(np.int32)
    for point in tqdm(points_to_be_assigned):
        current_lowest_neighbor = neighbor_gradient_dict[tuple(point)]
        current_point = point
        while current_lowest_neighbor is not None:
            current_point = current_lowest_neighbor
            if tuple(current_lowest_neighbor) in neighbor_gradient_dict:
                current_lowest_neighbor = neighbor_gradient_dict[tuple(current_lowest_neighbor)]
            else:
                break
        closest_index = nearest_indices[current_point[0], current_point[1], current_point[2]]
        wts[point[0], point[1], point[2]] = closest_index
        
    if not os.path.exists(predicted_label_path):
        os.makedirs(predicted_label_path)
    filepath = os.path.join(predicted_label_path, f"{inference_filename}_labels_.npy")
    np.save(filepath, wts)
    tifffile.imwrite(filepath, wts)
    print("file {filepath} saved, number of cells = {num_features}".format(filepath = filepath, num_features = wts.max()))


@njit
def create_label(image_input, locations, euclidian_distances_inverse):
    # Pad the image with zeros
    padded_image = image_input

    # Prepare output label array (same shape as padded image)
    label = np.empty_like(padded_image)
    image_shape = padded_image.shape

    # Iterate over the padded image
    for i in range(1, image_shape[0] - 1):
        for j in range(1, image_shape[1] - 1):
            for k in range(1, image_shape[2] - 1):
                # Extract 3x3x3 neighborhood values (excluding the boundary)
                all_values_neighborhood = np.array([padded_image[loc[0] + i, loc[1] + j, loc[2] + k] for loc in locations])
                
                # Compute the difference between the center and neighbors
                centre_minus_rest = padded_image[i, j, k] - all_values_neighborhood
                
                if np.all(centre_minus_rest < 0):  # Check if all neighbors are greater
                    label[i, j, k] = 13
                else:
                    centre_minus_rest_divided = centre_minus_rest * euclidian_distances_inverse
                    centre_minus_rest_divided[13] = -100  # Ignore the center value
                    class_label = np.argmax(centre_minus_rest_divided)
                    label[i, j, k] = class_label

    # Remove the padding from the result
    return label[1:-1, 1:-1, 1:-1]

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

from numba.typed import List

@njit
def compute_gradient_maps(trans_label, ellipsoid_locs, local_gradient, offsets, ellipsoid_list):
    for index in ellipsoid_locs:
        all_neighbour_values = []
        for i, j, k in offsets:
            neighbor = (index[0] + i, index[1] + j, index[2] + k)
            # Membership check using typed list
            is_valid = False
            for loc in ellipsoid_list:
                if loc[0] == neighbor[0] and loc[1] == neighbor[1] and loc[2] == neighbor[2]:
                    is_valid = True
                    break
            if is_valid:
                all_neighbour_values.append(trans_label[neighbor[0], neighbor[1], neighbor[2]])
            else:
                all_neighbour_values.append(0)  # Placeholder value for out-of-bound neighbors
        
        all_neighbour_values = np.array(all_neighbour_values)
        centre_minus_rest = trans_label[index[0], index[1], index[2]] - all_neighbour_values
        if np.all(centre_minus_rest >= 0):  # Check if reached local maximum
            local_gradient[index[0], index[1], index[2]] = 3
        else:
            centre_minus_rest_divided = centre_minus_rest  #* euclidian_distances_offsets
            centre_minus_rest_divided[3] = 1000  # Ignore the center value
            local_gradient[index[0], index[1], index[2]] = np.argmin(centre_minus_rest_divided)
    return local_gradient


@njit
def compute_gradient_maps_return_dict(trans_label, ellipsoid_locs, offsets, ellipsoid_list):
    location_dict = {n:[] for n in range(7)}
    for index in ellipsoid_locs:
        all_neighbour_values = []
        for i, j, k in offsets:
            neighbor = (index[0] + i, index[1] + j, index[2] + k)
            # Membership check using typed list
            is_valid = False
            for loc in ellipsoid_list:
                if loc[0] == neighbor[0] and loc[1] == neighbor[1] and loc[2] == neighbor[2]:
                    is_valid = True
                    break
            if is_valid:
                all_neighbour_values.append(trans_label[neighbor[0], neighbor[1], neighbor[2]])
            else:
                all_neighbour_values.append(0)  # Placeholder value for out-of-bound neighbors
        
        all_neighbour_values = np.array(all_neighbour_values)
        centre_minus_rest = trans_label[index[0], index[1], index[2]] - all_neighbour_values
        if np.all(centre_minus_rest >= 0):  # Check if reached local maximum
            local_gradient[index[0], index[1], index[2]] = 3
        else:
            centre_minus_rest_divided = centre_minus_rest  #* euclidian_distances_offsets
            centre_minus_rest_divided[3] = 1000  # Ignore the center value
            local_gradient[index[0], index[1], index[2]] = np.argmin(centre_minus_rest_divided)
    return local_gradient

from numba import float64, int32, int64, types, typed

num_images = 24

    
images, labels = load_training_images_and_labels("generated_data", 
                                                  num_images = num_images, 
                                                  image_format=".tif", 
                                                  label_format=".npy")
transformed_labels = images

offsets = np.array([
  [i, j, k]
  for i in range(-1, 2)
  for j in range(-1, 2)
  for k in range(-1, 2)
  if abs(i) + abs(j) + abs(k) <= 1
])

local_gradient_labels = []
euclidian_distances_offsets = np.linalg.norm(offsets, axis=1)
all_labels_closest_cells_per_cell = []

# before: 1min 26 for 24 images: 3.58 seconds per image

# after:

from concurrent.futures import ProcessPoolExecutor

def process_slices(slice_chunk, trans_label, integer_label, offsets):
    local_gradient_chunk = np.zeros_like(trans_label).astype(np.int32)
    for i, slice_tuple in enumerate(slice_chunk):
        if slice_tuple is not None:
            ellipsoid_locs = np.argwhere(integer_label[slice_tuple] == (i + 1)) + np.array([s.start for s in slice_tuple])
            if ellipsoid_locs.shape[0] > 40:
                relevant_values = trans_label[ellipsoid_locs[:, 0], ellipsoid_locs[:, 1], ellipsoid_locs[:, 2]]
                ellipsoid_list = List.empty_list(int64[:])

                for loc in ellipsoid_locs:
                    ellipsoid_list.append(loc)
                local_gradient_chunk = compute_gradient_maps(trans_label, ellipsoid_locs, local_gradient_chunk, offsets, ellipsoid_list)
    return local_gradient_chunk

def parallel_gradient_computation(labels, transformed_labels, offsets, num_cores=4):
    local_gradient_labels = []

    for i in tqdm(range(len(labels))):
        trans_label = transformed_labels[i] + 30.0
        integer_label = labels[i]
        slices = find_objects(integer_label)

        # Split slices into chunks for parallel processing
        slice_chunks = np.array_split(slices, num_cores)

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Map slice chunks to parallel workers
            results = list(executor.map(process_slices, slice_chunks, 
                                        [trans_label] * num_cores, 
                                        [integer_label] * num_cores, 
                                        [offsets] * num_cores))
            
        import pdb; pdb.set_trace()

        # Combine results from all workers
        combined_gradient = np.sum(results, axis=0)
        local_gradient_labels.append(combined_gradient)

    return local_gradient_labels

if __name__ == '__main__':

    local_gradient_labels = parallel_gradient_computation(labels, transformed_labels, offsets)

import pdb; pdb.set_trace()



for i in tqdm(range(len(labels))):
    trans_label = transformed_labels[i] + 30.0
    integer_label = labels[i]
    max_label = integer_label.max()
    local_gradient = np.ones_like(trans_label).astype(np.int32)
    local_gradient = local_gradient * 6
    slices = find_objects(integer_label)
    closest_cells_per_cell = []

    for i, slice_tuple in enumerate(slices):
        if slice_tuple is not None:  # Check if the slice is not None

            ellipsoid_locs = np.argwhere(integer_label[slice_tuple] == (i + 1)) + np.array([s.start for s in slice_tuple])
            if ellipsoid_locs.shape[0] > 40:
                relevant_values = trans_label[ellipsoid_locs[:, 0], ellipsoid_locs[:, 1], ellipsoid_locs[:, 2]]
                ellipsoid_list = List.empty_list(int64[:])

                for loc in ellipsoid_locs:
                    ellipsoid_list.append(loc)  # Convert each row to a tuple
                local_gradient = compute_gradient_maps(trans_label, ellipsoid_locs, local_gradient, offsets, ellipsoid_list)

    local_gradient_labels.append(local_gradient)

import pdb; pdb.set_trace()

import multiprocessing

print("computing gradient maps...")
import time
t0 = time.time()

for i in tqdm(range(len(labels))):
    trans_label = transformed_labels[i] + 30.0
    integer_label = labels[i]
    max_label = integer_label.max()
    local_gradient = np.ones_like(trans_label).astype(np.int32)
    local_gradient = local_gradient * 6
    slices = find_objects(integer_label)
    closest_cells_per_cell = []

    for i, slice_tuple in enumerate(slices):
        if slice_tuple is not None:  # Check if the slice is not None

            ellipsoid_locs = np.argwhere(integer_label[slice_tuple] == (i + 1)) + np.array([s.start for s in slice_tuple])
            if ellipsoid_locs.shape[0] > 40:
                relevant_values = trans_label[ellipsoid_locs[:, 0], ellipsoid_locs[:, 1], ellipsoid_locs[:, 2]]
                ellipsoid_list = List.empty_list(int64[:])

                for loc in ellipsoid_locs:
                    ellipsoid_list.append(loc)  # Convert each row to a tuple
                local_gradient = compute_gradient_maps(trans_label, ellipsoid_locs, local_gradient, offsets, ellipsoid_list)

    local_gradient_labels.append(local_gradient)

t1 = time.time()

total_time = t1-t0

print("total_time", total_time)
import pdb; pdb.set_trace()
    
local_gradient_labels = np.array(local_gradient_labels)#np.array(local_gradient_labels)
