import numpy as np
import os
import cv2
import pickle

def process_directory(directory):
    folder_index = {}
    result_indices = []
    result_folders = []

    index = 0
    for root, dirs, files in os.walk(directory):
        for directory_name in dirs:
            if directory_name not in folder_index:
                folder_index[directory_name] = index
                index += 1

    for root, dirs, files in os.walk(directory):
        for file_name in files:
            folder_name = os.path.basename(root)
            folder_value = folder_index[folder_name]
            result_indices.append(folder_value)
            result_folders.append(folder_name)

    return result_indices, result_folders
    
def load_images_from_folder(folder, num_images):
    images = []
    count = 0

    # Iterate over all subdirectories and files in the given folder
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if count >= num_images:
                break
            img = cv2.imread(os.path.join(root, filename))
            if img is not None:
                images.append(img)
                count += 1

    return images

# Example usage:
directory_path = 'training'
result_indices, result_folders = process_directory(directory_path)

print('Labels:')
print(result_indices)
print('Label names:')
print(result_folders)

if not os.path.exists(directory_path+'_packed'):
    os.makedirs(directory_path+'_packed')

output_file_int = directory_path+'_packed/labels.pkl'
with open(output_file_int, 'wb') as f:
    pickle.dump(result_indices, f)

output_file_names = directory_path+'_packed/label_names.pkl'
with open(output_file_names, 'wb') as f:
    pickle.dump(result_folders, f)

num_images = len(result_indices)
# Load images from the folder
print("Loading images...")
images = load_images_from_folder(directory_path, num_images)
# Reshape the images to 250x250 pixels
images_resized = [cv2.resize(image, (250, 250)) for image in images]
# Convert the image list to a NumPy array
images_array = np.array(images_resized)

output_file_int = directory_path+'_packed/images.pkl'
with open(output_file_int, 'wb') as f:
    pickle.dump(images_array, f)