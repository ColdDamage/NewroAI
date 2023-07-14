import numpy as np
import Newro
import os
import cv2

def load_images_from_folder(folder, num_images):
    images = []
    count = 0
    for filename in os.listdir(folder):
        if count >= num_images:
            break
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            count += 1
    return images

# Model and variables
model_name = "flower_network_model"
train_folder = "train"
num_images = 400
# initialize model instance
model = Newro.Model()
# Define layers (weights, layers, activation type)
# input layer
model.addLayer(250 * 250 * 3, 64, Newro.Activation_ReLU())
# hidden layer
model.addLayer(64, 64, Newro.Activation_ReLU())
# output layer
model.addLayer(64, 2, Newro.Activation_Softmax())

# Training loop with batch training
batch_size = 8
epochs = 50

# Load images from the "train" folder
images = load_images_from_folder(train_folder, num_images)
# Assign labels to the images
labels = np.zeros(num_images)
labels[200:] = 1
y = labels.astype(int)  # Convert the labels to integers
# Reshape the images to 250x250 pixels
images_resized = [cv2.resize(image, (250, 250)) for image in images]
# Convert the image list to a NumPy array
X = np.array(images_resized)
# Normalize the image data
X = X.astype('float32') / 255.0
# Reshape the input data
X = X.reshape(num_images, -1)

steps = len(X) // batch_size

# Create loss function object
loss_function = Newro.Loss_CategoricalCrossentropy()

for epoch in range(epochs):
    epoch_loss = 0.0

    for step in range(steps):
        batch_start = step * batch_size
        batch_end = (step + 1) * batch_size
        X_batch = X[batch_start:batch_end]
        y_batch = y[batch_start:batch_end]

        # Forward pass
        output = model.forward_Pass(X_batch)
        
        #Loss calculation
        loss = loss_function.calculate(output, y_batch)
        epoch_loss += loss

        # Backward pass
        model.backward_Pass(output, y_batch)

    # Print average loss for the epoch
    average_loss = epoch_loss / steps
    print("Epoch:", epoch, "Loss:", average_loss)

# save trained model
model.save_Model(model_name)