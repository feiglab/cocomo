import mdtraj as md
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import scipy
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='traj', help='Input trajectory file', required=True)
parser.add_argument('-p', dest='top', help='Topology file (psf/pdb)', required=True)
parser.add_argument('-n', dest='n', help='Number of monomers', required=True, type=int)
parser.add_argument('-res', dest='res', help='Number of residues in a monomer', required=True, type=int)
parser.add_argument('-b', dest='box', help='Box size', required=True, type=float)
parser.add_argument('--cutoff_factor', dest='cutoff_factor', help='Cutoff factor for distance calculation', default=2.4, type=float)
parser.add_argument('--low_threshold', dest='low_threshold', help='Lower threshold for cluster size', default=5, type=int)
parser.add_argument('--high_threshold', dest='high_threshold', help='Higher threshold for cluster size', default=50, type=int)
args = parser.parse_args()

# Load trajectory and initialize variables
cluster = md.load(args.traj, top=args.top)
n = args.n
box = args.box
res = args.res
cutoff_factor = args.cutoff_factor
low_threshold = args.low_threshold
high_threshold = args.high_threshold

# Compute centers of monomers
center = np.average(np.reshape(cluster.xyz, (cluster.n_frames, n, res, 3)), axis=2)
center_traj = cluster.atom_slice(np.arange(n, dtype=int))
center_traj.xyz = center
center_traj.unitcell_lengths = cluster.unitcell_lengths
center_traj.unitcell_angles = cluster.unitcell_angles

# Determine cutoff distance based on radius of gyration
cut_off = np.average(md.compute_rg(cluster.atom_slice(np.arange(res, dtype=int)))) * cutoff_factor

# Initialize lists for cluster sizes and concentration calculations
cluster_size_list = []
conc_below_low, conc_between, conc_above_high = [], [], []

# Loop through each frame to compute cluster sizes
for j in range(cluster.n_frames):
    print(f"Processing frame {j + 1}/{cluster.n_frames}")

    x = np.arange(n, dtype=int)
    X, Y = np.meshgrid(x, x)
    atom_pairs = np.column_stack((X.flatten(), Y.flatten()))

    distances = md.compute_distances(center_traj[j], atom_pairs, periodic=True)
    Z = single(scipy.spatial.distance.squareform(distances.reshape(n, n)))
    _, cluster_sizes = np.unique(fcluster(Z, cut_off, criterion='distance'), return_counts=True)

    cluster_size_list.append(cluster_sizes.tolist())

    # Calculate concentrations for different cluster size ranges
    tmp = np.array(cluster_sizes)
    conc_below_low.append((np.sum(tmp[tmp < low_threshold]) / n) * (n / (0.0006022 * (box ** 3))))
    conc_between.append((np.sum(tmp[(tmp >= low_threshold) & (tmp <= high_threshold)]) / n) * (n / (0.0006022 * (box ** 3))))
    conc_above_high.append((np.sum(tmp[tmp > high_threshold]) / n) * (n / (0.0006022 * (box ** 3))))

# Save concentration data with column descriptions
with open("conc.dat", "w") as f:
    f.write("# Columns: Concentration below low_threshold, Concentration between thresholds, Concentration above high_threshold\n")
    np.savetxt(f, np.column_stack((conc_below_low, conc_between, conc_above_high)), delimiter=" ", fmt='%s')

# Flatten cluster sizes and compute histogram
all_cluster_sizes = [size for frame_sizes in cluster_size_list for size in frame_sizes]
a, b = np.histogram(all_cluster_sizes, bins=np.linspace(0.5, n + 0.5, n + 1))

# Save histogram data with column descriptions
size = np.linspace(1, n, n)
weighted_cluster_size = size * a / (n * cluster.n_frames)
with open("hist.dat", "w") as f:
    f.write("# Columns: Cluster size, Weighted cluster size\n")
    np.savetxt(f, np.column_stack((size, weighted_cluster_size)), delimiter=" ", fmt='%s')

# Compute and print average solute concentration
average_solute_concentration = np.mean(conc_below_low)
print(f"Average solute concentration (below low_threshold, in mM): {average_solute_concentration}")

print("Processing complete. Data saved to conc.dat and hist.dat.")
