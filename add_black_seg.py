import numpy as np
import os

# Define the path to the directory containing your npz files
npz_dir = '.\\difnpz_healthy_only_t1ce_64_nk\\healthy_07'

new_path = '.\\difnpz_healthy_only_t1ce_64_nk\\healthy_07'

if not os.path.exists(new_path):
    os.makedirs(new_path)

# Loop over each npz file in the directory
for filename in os.listdir(npz_dir):
    # Load the original data from the npz file
    data = np.load(os.path.join(npz_dir, filename))
    orig_array = data["t1ce"]
    # Add a new channel of zeros to the original array
    new_array = np.zeros((64, 64, 2))
    new_array[:, :, 0] = orig_array
    # Save the new array as a new npz file
    np.savez_compressed(os.path.join(new_path, filename),
                        t1ce=new_array[:, :, 0],
                        # ,flair=flair_img,  # Todo: change here
                        seg=new_array[:, :, 1]
                        )