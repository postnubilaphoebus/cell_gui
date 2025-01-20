# label transform constants
label_transform_mode = "3D" # 3D_to_2D or 2D are also options
high_value = 20.0 
low_value = -20.0 
background_maximum = -5.0 # -5.0
foreground_minimum = -2.0 # - 2.0
cell_decay_base = 20.0#20.0 # 15.0
distance_transform_constant = 0.1 
ignore_index = -100

# model constants
learning_rate = 0.001 
betas = (0.9, 0.99) 
conv_channels = (32, 64, 128, 256, 512) 
load_pretrained = False
model_weights_path = "best_model.pth"
maximum_epochs = 700 # 700
width_multiplier = 1

## learning rate scheduler constants
base_lr = 0.000001 
max_lr = 0.01 
step_size_up = 400 
step_size_down = 250

# training and inference constants
verbosity_flag = True
use_masks = False
data_augmentation_types = ["rotate",
                           "random_gamma",
                           "brightness_shift",  
                           "motion_blur",
                           "gaussian_noise"]
# "rotate",

num_images = 6
num_validation_images = 1
mini_batch_size = 1
early_stopping_patience = 200
training_folder = "training_data"#"training_data" # c_elegans_train
inference_folder = "inference_data"
background_path = "background_images"
inference_resolution_upsampling = (2.285, 1, 1)#(2, 1, 1)#(2, 1, 1)# None# 
image_shape_dim = 64 # (will be repeated number_of_dim times because we only process isotropic blocks) # 64

# post-processing
postprocessing_background_threshold = 0.07# 0.15 # 0.09
post_processing_centre_threshold = 0.25 #0.25 #0.15 # 0.35 # 0.2
minimum_cell_size = 200
minimum_cell_size_boundary = 60
min_distance_between_cells = 2
extra_padding_width = 15
