import numpy as np
import Newro
import signal
import sys
import pickle
from tqdm import tqdm

# Training variables
model_name = "model_name" # Model name for saving model
label_file = 'training_packed/labels.pkl' # File path for labels file
images_file = 'training_packed/images.pkl' # File path for images file
batch_size = 8 # Batches for training
epochs = 50 # Epochs for training

# initialize model instance
model = Newro.Model()
# Define layers (weights, layers, activation type)
print("Creating layers...")
# input layer
model.addLayer(250 * 250 * 3, 64, Newro.Activation_ReLU())
# hidden layer
model.addLayer(64, 64, Newro.Activation_ReLU())
# output layer
model.addLayer(64, 2, Newro.Activation_Softmax())

print("Loading label array...")
# Read the labels from the pickle file
with open(label_file, 'rb') as f:
    label_result_list = pickle.load(f)
labels = np.array(label_result_list)
y = labels.astype(int)  # Convert the labels to integers

print("Loading image array...")
# Read the images from the pickle file
with open(images_file, 'rb') as f:
    image_result_list = pickle.load(f)
images = np.array(image_result_list)

# Normalize the image data
X = images.astype('float32') / 255.0
# Reshape the input data
X = X.reshape(len(images), -1)

steps = len(X) // batch_size

# Create loss function object
loss_function = Newro.Loss_CategoricalCrossentropy()

def cleanup():
    # Perform cleanup actions here
    print("Saving model...")
    model.save_Model(model_name+"_interrupt")

def signal_handler(sig, frame):
    # Handle the KeyboardInterrupt signal (Ctrl+C)
    print("KeyboardInterrupt received!")
    cleanup()
    sys.exit(0)

# Register the signal handler for KeyboardInterrupt
signal.signal(signal.SIGINT, signal_handler)

# Training loop for epochs or until keyboardinterrupt
print("Training model...")
try:
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for step in tqdm(range(steps), leave=False):
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

except KeyboardInterrupt:
    cleanup()
    sys.exit(0)

# save trained model
print("Saving model "+model_name)
model.save_Model(model_name)