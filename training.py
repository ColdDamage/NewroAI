import numpy as np
from Newro import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss, Loss_CategoricalCrossentropy
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

model_name = 'flower_network_model.npz'
train_folder = "train"
num_images = 400

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

# Define layers
# input layer
dense1 = Layer_Dense(250 * 250 * 3, 128)
activation1 = Activation_ReLU()
# hidden layer
dense2 = Layer_Dense(128, 128)
activation2 = Activation_ReLU()
# output layer
dense3 = Layer_Dense(128, 2)
activation3 = Activation_Softmax()

# Create loss function object
loss_function = Loss_CategoricalCrossentropy()

# Training loop with batch training
batch_size = 8
epochs = 50
steps = len(X) // batch_size

for epoch in range(epochs):
    epoch_loss = 0.0

    for step in range(steps):
        batch_start = step * batch_size
        batch_end = (step + 1) * batch_size
        X_batch = X[batch_start:batch_end]
        y_batch = y[batch_start:batch_end]

        # Forward pass
        dense1.forward(X_batch)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        activation3.forward(dense3.output)
        loss = loss_function.calculate(activation3.output, y_batch)
        epoch_loss += loss

        # Backward pass
        loss_function.backward(activation3.output, y_batch)
        activation3.backward(loss_function.dinputs)
        dense3.backward(activation3.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        dense1.update_parameters(0.05)
        dense2.update_parameters(0.05)
        dense3.update_parameters(0.05)

    # Print average loss for the epoch
    average_loss = epoch_loss / steps
    print("Epoch:", epoch, "Loss:", average_loss)

# Define trained model with current weights and biases
trained_model = {
    'dense1_weights': dense1.weights,
    'dense1_biases': dense1.biases,
    'dense2_weights': dense2.weights,
    'dense2_biases': dense2.biases,
    'dense3_weights': dense3.weights,
    'dense3_biases': dense3.biases
}

# Save current weights and biases to a file
def save_model(model, file_path):
    np.savez(file_path, **model)
    
save_model(trained_model, model_name)