import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import gaussian_filter
from detect_ellipsoid_coll import algebraic_separation_condition
from tqdm import tqdm
from scipy.stats import skewnorm
import random

class Ellipsoid:
    def __init__(self, position, semi_axes, velocity, orientation):
        self.position = np.array(position)
        self.semi_axes = np.array(semi_axes)
        self.velocity = np.array(velocity)
        self.orientation = np.array(orientation)

def detect_ellipsoid_collision(e1, e2):
    dist = max(np.abs(e1.position - e2.position))
    return dist < (e1.semi_axes.max() + e2.semi_axes.max())

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


def generate_random_orientation(max_angle_degrees):
    max_angle_radians = np.radians(max_angle_degrees)
    angle = np.random.uniform(0, max_angle_radians)
    axis = np.random.choice(['x', 'y'])

    if axis == 'x':
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

def apply_center_pull(ellipsoid, center, strength=0.01):
    direction_to_center = center - ellipsoid.position
    direction_to_center /= np.linalg.norm(direction_to_center)
    ellipsoid.velocity += direction_to_center * strength

def lubachevsky_stillinger_ellipsoids(positions, initial_semi_axes, growth_rate, max_steps, max_semi_axes, max_orientation_angle):
    ellipsoids = []
    num_ellipsoids = len(positions)
    pause_growth_interval = 2
    volume_limit = 63
    center = np.array([volume_limit / 2] * 3)

    for position in positions:
        velocity = np.random.uniform(-1, 1, size=3)
        orientation = generate_random_orientation(max_orientation_angle)
        ellipsoids.append(Ellipsoid(position, initial_semi_axes, velocity, orientation))

    for step in tqdm(range(max_steps)):
        if step % pause_growth_interval == 0:
            for ellipsoid in ellipsoids:
                growth = growth_rate * (1 + np.random.uniform(-0.7, 0.7, size=3))
                ellipsoid.semi_axes = np.minimum(ellipsoid.semi_axes + growth, max_semi_axes)

        for i in range(num_ellipsoids):
            for j in range(i + 1, num_ellipsoids):
                if detect_ellipsoid_collision(ellipsoids[i], ellipsoids[j]):
                    if algebraic_separation_condition(ellipsoids[i].semi_axes, 
                                                  ellipsoids[j].semi_axes, 
                                                  ellipsoids[i].position, 
                                                  ellipsoids[j].position, 
                                                  ellipsoids[i].orientation, 
                                                  ellipsoids[j].orientation): 
                        resolve_ellipsoid_collision(ellipsoids[i], ellipsoids[j])

        for ellipsoid in ellipsoids:
            apply_center_pull(ellipsoid, center)
            ellipsoid.position += ellipsoid.velocity * np.abs(np.mean(growth_rate))
            for k in range(3):
                if ellipsoid.position[k] < -3 or ellipsoid.position[k] > 67:
                    direction_to_center = center - ellipsoid.position
                    ellipsoid.velocity = direction_to_center / np.linalg.norm(direction_to_center)
                    ellipsoid.position[k] = -3
    #print("resolving final collisions...")
    new_growth_rate = np.array([0.01, 0.01, 0.01])
    collision_detected = True
    oob = True
    while collision_detected or oob:
        collision_detected = False
        oob = False
        for i in range(num_ellipsoids):
            for j in range(i + 1, num_ellipsoids):
                if detect_ellipsoid_collision(ellipsoids[i], ellipsoids[j]):
                    if algebraic_separation_condition(ellipsoids[i].semi_axes, 
                                                  ellipsoids[j].semi_axes, 
                                                  ellipsoids[i].position, 
                                                  ellipsoids[j].position, 
                                                  ellipsoids[i].orientation, 
                                                  ellipsoids[j].orientation): 
                        resolve_ellipsoid_collision(ellipsoids[i], ellipsoids[j], always_resolve=True)
                        collision_detected = True
        for ellipsoid in ellipsoids:
            ellipsoid.position += ellipsoid.velocity * new_growth_rate
            for k in range(3):
                if ellipsoid.position[k] < -3 or ellipsoid.position[k] > 67:
                    direction_to_center = center - ellipsoid.position
                    ellipsoid.velocity = direction_to_center / np.linalg.norm(direction_to_center)
                    ellipsoid.position[k] = -3
                    oob = True

    return ellipsoids

def is_point_inside_ellipsoid(points, ellipsoid):
    translated_points = points - ellipsoid.position
    rotated_points = translated_points @ ellipsoid.orientation.T
    normalized_points = rotated_points / ellipsoid.semi_axes
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
    x, y, z = np.meshgrid(np.arange(volume_size), np.arange(volume_size), np.arange(volume_size), indexing='ij')
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    volume = np.zeros((volume_size, volume_size, volume_size), dtype=np.uint16)
    label_volume = np.zeros((volume_size, volume_size, volume_size), dtype=np.uint16)

    for idx, ellipsoid in enumerate(ellipsoids, start=1):
        inside = is_point_inside_ellipsoid(points, ellipsoid)
        volume.ravel()[inside] = 1
        label_volume.ravel()[inside] = idx

    label_volume = pack_labels(label_volume)
    print("num cells", max(label_volume.ravel()))
    return volume, label_volume

def compute_volume(semi_axes):
    return 4/3 * np.pi * semi_axes[0] * semi_axes[1] * semi_axes[2]

def compute_density(ellipsoids, volume):
    total_volume = sum(compute_volume(e.semi_axes) for e in ellipsoids)
    return total_volume / volume

def generate_synthetic_volumes(label_file, random_seed, max_steps):

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    data = skewnorm.rvs(1, loc=200, scale=50, size=1000)
    data = np.clip(data, 100, 300)  # Clip the data to stay within the desired range
    num_cells = round(np.random.choice(data))

    # Parameters
    volume_size = 64
    initial_semi_axes = np.array([2.0, 2.0, 2.0])
    growth_rate = np.array([0.01, 0.01, 0.01])
    max_semi_axes = np.array([12, 12, 12])
    max_orientation_angle = 60  # Maximum angle in degrees from the xy-plane

    # Load positions from file and apply random variation
    example_file = label_file
    example_file[example_file == -1] = 0
    example_file = random_rotate_3d_cube(example_file)
    max_index = max(example_file.ravel())
    positions = []
    for i in range(1, max_index + 1):
        locations = np.where(example_file == i)
        if len(locations[0]) == 0:
            continue  
        cm = np.mean(locations, axis=1)
        positions.append(cm)

    if num_cells < len(positions):
        n_remove = len(positions) - num_cells
        indices_to_remove = random.sample(range(len(positions)), n_remove)
        indices_to_remove.sort(reverse=True)
        for index in indices_to_remove:
            del positions[index]

    positions = np.array(positions)
    positions += np.random.uniform(-0.8, 0.8, positions.shape)
    ellipsoids = lubachevsky_stillinger_ellipsoids(positions, initial_semi_axes, growth_rate, max_steps, max_semi_axes, max_orientation_angle)
    volume, label_volume = generate_3d_volume(ellipsoids, volume_size)
    blurred_volume = gaussian_filter(volume.astype(np.float32), sigma=(5, 1, 1))
    noisy_blurred_volume = blurred_volume + np.random.normal(0, 0.1, size=blurred_volume.shape)
    for idx in range(1, np.max(label_volume) + 1):
        ellipsoid_mask = (label_volume == idx)
        # mean = 0  # Mean of the underlying normal distribution
        # sigma = 6  # Standard deviation of the underlying normal distribution

        # # Sample from the log-normal distribution
        # log_normal_sample = np.random.lognormal(mean, sigma)

        # # Scale the log-normal sample to your desired range [0.6, 1.4]
        # low, high = 0.6, 3
        # adjustment_factor = low + (high - low) * (log_normal_sample / (log_normal_sample + 1))
        adjustment_factor = np.random.uniform(0.6, 1.5)  # Adjust the range as needed
        noisy_blurred_volume[ellipsoid_mask] *= adjustment_factor
    for idx in range(1, np.max(label_volume) + 1):
        ellipsoid_mask = (label_volume == idx)
        ellipsoid_values = noisy_blurred_volume[ellipsoid_mask]
        adjustment_factor = np.random.uniform(0, 0.1)
        mean_intensity = np.mean(ellipsoid_values)
        mean_diff = np.mean(np.abs(np.diff(ellipsoid_values)))
        adjusted_values = np.random.normal(loc=mean_intensity, scale=mean_diff, size=ellipsoid_values.shape)
        noisy_blurred_volume[ellipsoid_mask] = adjustment_factor * adjusted_values + (1 - adjustment_factor) * ellipsoid_values

    #noisy_blurred_volume = (noisy_blurred_volume - np.min(noisy_blurred_volume)) / (np.max(noisy_blurred_volume) - np.min(noisy_blurred_volume))
    #tifffile.imwrite(img_name+'.tif', noisy_blurred_volume)
    #np.save(img_name + '.npy', label_volume)
    return noisy_blurred_volume, label_volume

# import skimage

# image = skimage.io.imread("bottom_right_upsampled.tif")
# label = np.load("bottom_right_upsampledtif_mask.npy")
# label[label == -1] = 0
# # label[label > 0] = 1

# image2 = skimage.io.imread("bottom_left_upsampled.tif")
# label2 = np.load("bottom_left_upsampledtif_mask.npy")
# label2[label2 == -1] = 0
# # label2[label2 > 0] = 1

# image3 = skimage.io.imread("mid_section_upsampled.tif")
# label3 = np.load("mid_section_upsampledtif_mask.npy")
# label3[label3 == -1] = 0
# # label3[label3 > 0] = 1

# image4 = skimage.io.imread("bad_lighting_upsampled.tif")
# label4 = np.load("bad_lighting_upsampledtif_mask.npy")
# label4[label4 == -1] = 0
# # label4[label4 > 0] = 1

# image5 = skimage.io.imread("cut_again_upsampled.tif")
# label5 = np.load("cut_again_upsampledtif_mask.npy")
# label5[label5 == -1] = 0
# # label5[label5 > 0] = 1

# image6 = skimage.io.imread("cut_again_img3_upsampled.tif")
# label6 = np.load("cut_again_img3_upsampledtif_mask.npy")
# label6[label6 == -1] = 0
# # label6[label6 > 0] = 1

# images = [image, image2, image3, image4, image5, image6]
# labels = [label, label2, label3, label4, label5, label6]

# volume_size = 64

# for i in range(50):
#     noisy_blurred_volume, label_volume = generate_synthetic_volumes(labels[i%4], random_seed=i, max_steps=650)

#     noisy_blurred_volume = (noisy_blurred_volume - np.min(noisy_blurred_volume)) / (np.max(noisy_blurred_volume) - np.min(noisy_blurred_volume))

#     tifffile.imwrite("noisy"+str(i) + '.tif', noisy_blurred_volume)

#     np.save("noisy"+str(i) + '.npy', label_volume)

    # # Plot slices
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # slices = [noisy_blurred_volume[volume_size // 4, :, :], noisy_blurred_volume[volume_size // 2, :, :], noisy_blurred_volume[3 * volume_size // 4, :, :]]

    # for i, ax in enumerate(axes):
    #     ax.imshow(slices[i], cmap='gray')
    #     ax.set_title(f'Slice {i + 1}')

    # plt.show()
