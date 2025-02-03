import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

# Define the colormap and create a discrete version
cmap = plt.get_cmap('viridis')
n_colors = 1000
cmap_discrete = cmap(np.linspace(0, 1, n_colors))

# Define the range for the color bars
ranges = {
    'bar1': (-0.2, 0.6),
    'bar2': (-0.9, 0.7),
    'bar3': (-0.7, -0.5)
}

# Create a function to get the color indices from the discrete colormap
def get_color_indices(vmin, vmax):
    idx_min = int((1 + vmin) * (n_colors - 1) / 2)  # Mapping vmin to colormap index
    idx_max = int((1 + vmax) * (n_colors - 1) / 2)  # Mapping vmax to colormap index
    return cmap_discrete[idx_min:idx_max]  # Get the color range

# Create a figure
fig = plt.figure(figsize=(10, 6))

# Plot the full range color bar
full_bar_height = 0.05  # Uniform height for all bars
ax_full = fig.add_axes([0.05, 0.8, 0.9, full_bar_height])  # Adjusted width and centering
full_color_map = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=-1, vmax=1))
full_color_map.set_array([])
cbar_full = ColorbarBase(ax_full, cmap='viridis', norm=plt.Normalize(vmin=-1, vmax=1),
                         orientation='horizontal')
cbar_full.ax.set_xlabel('Range: -1 to 1')
ax_full.text(0.5, 1.2, 'Inference Image', ha='center', va='center', transform=ax_full.transAxes, fontsize=12)

# Plot the smaller color bars using the correct indices from the full colormap
y_positions = [0.6, 0.4, 0.2]  # Y positions for the smaller color bars
labels = ['Training Augmentation 1', 'Training Augmentation 2', 'Training Augmentation 3']  # Labels for smaller color bars

for i, (label, (vmin, vmax)) in enumerate(ranges.items()):
    # Get the colors directly from cmap_discrete based on vmin and vmax
    color_starts = get_color_indices(vmin, vmax)
    
    # Create a listed colormap from the selected color indices
    color_map = mcolors.ListedColormap(color_starts)
    
    # Calculate the normalized position of vmin and vmax relative to [-1, 1]
    x_start = (1 + vmin) / 2 * 0.9 + 0.05  # Start position of the color bar, scaled to 0.9 width
    x_width = (vmax - vmin) / 2 * 0.9      # Width of the color bar, scaled to 0.9 width
    
    # Create an axis for each color bar, positioned at the calculated x_start and x_width
    cbar_ax = fig.add_axes([x_start, y_positions[i], x_width, full_bar_height])  # Match height to full bar
    cbar = ColorbarBase(cbar_ax, cmap=color_map, orientation='horizontal', norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    cbar.ax.set_xlabel(f'Range: {vmin} to {vmax}')  # Label with the range on the bar
    
    # Add the text labels above the smaller color bars
    cbar_ax.text(0.5, 1.2, labels[i], ha='center', va='center', transform=cbar_ax.transAxes, fontsize=12)

plt.tight_layout()
plt.show()
