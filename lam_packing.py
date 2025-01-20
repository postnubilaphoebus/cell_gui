import numpy as np
import matplotlib.pyplot as plt
from lammps import lammps
from matplotlib.patches import Ellipse
from ctypes import *

# Constants
N_ELLIPSES = 50
DOMAIN_SIZE = 100
MIN_AXIS = 2
MAX_AXIS = 8
# Area of ellipses
def compute_packing_fraction(axes, domain_size):
  total_area = np.sum(np.pi * axes[:, 0] * axes[:, 1])
  return total_area / (domain_size ** 2)

# Visualize ellipses
def visualize_ellipses(positions, axes, domain_size, title, packing_fraction):
  fig, ax = plt.subplots(figsize=(8, 8))
  for pos, axis in zip(positions, axes):
    ellipse = Ellipse(
        xy=pos,
        width=axis[0],
        height=axis[1],
        edgecolor="blue",
        fill=False,
        linewidth=1.5,
    )
    ax.add_artist(ellipse)
  ax.set_xlim(0, domain_size)
  ax.set_ylim(0, domain_size)
  ax.set_aspect("equal")
  ax.set_title(f"{title}\nPacking Fraction: {packing_fraction:.3f}")
  plt.show()


def generate_random_ellipsoids(n, domain_size, min_axis, max_axis, max_attempts=1000):
    positions = []
    axes = []
    atom_types = []  # Keep track of atom types based on axes

    for _ in range(n):
        for attempt in range(max_attempts):
            # Generate random position and axes (for 3D ellipsoids)
            position = np.random.uniform(0, domain_size, size=(3,))
            axis = np.random.uniform(min_axis, max_axis, size=(3,))
            
            # Check overlap with existing ellipsoids
            overlap = False
            for existing_pos, existing_axis in zip(positions, axes):
                # Calculate the Euclidean distance between the centers of the ellipsoids
                distance = np.linalg.norm(position - existing_pos)
                # Compare with the sum of the radii of the ellipsoids (axes)
                max_axes = np.max(axis)
                max_existing_axes = np.max(existing_axis)
                if distance < (max_axes + max_existing_axes) / 2:
                    overlap = True
                    break

            if not overlap:
                positions.append(position)
                axes.append(axis)
                atom_types.append(len(set([tuple(a) for a in axes])))  # Assign atom type based on axis shape
                break
        else:
            raise RuntimeError(f"Could not place all {n} ellipsoids without overlap after {max_attempts} attempts.")

    return np.array(positions), np.array(axes), atom_types


def setup_lammps_simulation(positions, axes, atom_types, domain_size):
    lmp = lammps()

    # Get the number of atom types (unique types based on shape)
    num_atom_types = len(set(atom_types))

    # Initialize LAMMPS commands
    lmp.command("units lj")
    lmp.command("dimension 2")
    lmp.command("boundary p p p")
    lmp.command("atom_style ellipsoid")
    lmp.command(f"region box block 0 {domain_size} 0 {domain_size} -0.1 0.1")
    lmp.command(f"create_box {num_atom_types} box")  # Number of atom types should match unique atom types

    # Create ellipsoids and assign atom types
    for i, (pos, axis, atom_type) in enumerate(zip(positions, axes, atom_types)):
        lmp.command(f"create_atoms {atom_type} single {pos[0]} {pos[1]} 0.0")
        lmp.command(f"set atom {i + 1} shape {axis[0]} {axis[1]} 1")

    # Define pair style with specific parameters for each atom type
    lmp.command("pair_style gayberne 1.0 1.0 1.0 1.0")
    
    # Set pair coefficients for each unique pair of atom types
    unique_atom_types = sorted(set(atom_types))  # Get unique atom types in sorted order
    for i in unique_atom_types:
        for j in unique_atom_types:
            if i <= j:  # Pair coefficients are symmetric
                lmp.command(f"pair_coeff {i} {j} 1.0 1.7 1.7 3.4 3.4 1.0 1.0 1.0")

    # Define dynamics and simulation settings
    lmp.command("fix 1 all nve/limit 0.01")
    lmp.command("thermo 10")
    lmp.command("run 5000")

    # Extract optimized positions
    natoms = lmp.get_natoms()
    n3 = 3 * natoms
    x = (n3 * c_double)()  # Create array to store the atom coordinates

    # Use LAMMPS to fetch atom coordinates
    lmp.gather_atoms("x", 1, 3)  # This fills `x` with the atomic positions in a flat array

    # Extract coordinates manually using the array `x`
    optimized_positions = []
    for i in range(natoms):
        atom_x = x[3*i]    # x-coordinate of atom i
        atom_y = x[3*i+1]  # y-coordinate of atom i
        atom_z = x[3*i+2]  # z-coordinate of atom i
        optimized_positions.append([atom_x, atom_y, atom_z])

    lmp.close()
    return np.array(optimized_positions)

# lmp = lammps()
# print("lmp.version()", lmp.version())
n = 10
domain_size = 100
min_axis = 1
max_axis = 5

positions, axes, atom_types = generate_random_ellipsoids(n, domain_size, min_axis, max_axis)
optimized_positions = setup_lammps_simulation(positions, axes, atom_types, domain_size)
print("Optimized positions:\n", optimized_positions)

# # # Main Workflow
# positions, axes = generate_random_ellipses(N_ELLIPSES, DOMAIN_SIZE, MIN_AXIS, MAX_AXIS)
# # initial_packing_fraction = compute_packing_fraction(axes, DOMAIN_SIZE)

# # # Visualize initial state
# # visualize_ellipses(positions, axes, DOMAIN_SIZE, "Initial Packing", initial_packing_fraction)

# # # Optimize packing
# optimized_positions = setup_lammps_simulation(positions, axes, DOMAIN_SIZE)
# # optimized_packing_fraction = compute_packing_fraction(axes, DOMAIN_SIZE)

# # # Visualize optimized state
# # visualize_ellipses(optimized_positions, axes, DOMAIN_SIZE, "Optimized Packing", optimized_packing_fraction)
