import cv2
import numpy as np
from Newro import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss, Loss_CategoricalCrossentropy

model_name = 'flower_network_model.npz'

# Load the saved model
loaded_model = np.load(model_name, allow_pickle=True)

# Create layer instances and assign loaded weights and biases
# input layer
dense1 = Layer_Dense(loaded_model['dense1_weights'].shape[0], loaded_model['dense1_weights'].shape[1])
dense1.weights = loaded_model['dense1_weights']
dense1.biases = loaded_model['dense1_biases']
# hidden layer
dense2 = Layer_Dense(loaded_model['dense2_weights'].shape[0], loaded_model['dense2_weights'].shape[1])
dense2.weights = loaded_model['dense2_weights']
dense2.biases = loaded_model['dense2_biases']
# output layer
dense3 = Layer_Dense(loaded_model['dense3_weights'].shape[0], loaded_model['dense3_weights'].shape[1])
dense3.weights = loaded_model['dense3_weights']
dense3.biases = loaded_model['dense3_biases']

activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_Softmax()

while True:
    # Load the image to test
    print("\nInput image filename from /test_images to test, or type 0 to end")
    image = input()
    if image == "0":
        break
    
    image_path = "test_images/"+image
    test_image = cv2.imread(image_path)

    # Resize the test image to match the input size of the model
    resized_image = cv2.resize(test_image, (250, 250))

    # Convert the resized image to a NumPy array
    test_input = np.array([resized_image])

    # Normalize the test image data
    test_input = test_input.astype('float32') / 255.0

    # Reshape the test input data
    test_input = test_input.reshape(len(test_input), -1)

    # Forward pass
    dense1.forward(test_input)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    # Get the predicted class probabilities
    prediction = activation3.output[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # Print the prediction and confidence
    class_labels = ['Human (0)', 'Flower (1)']
    print("Prediction:", class_labels[predicted_class])
    print("Confidence:", confidence)