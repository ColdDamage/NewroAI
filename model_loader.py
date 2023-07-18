import cv2
import numpy as np
import Newro
import pickle

print("\nThis program loads a model from the folder you designate.\ne.g. to load the model \"example_model/example_model.npz\" just type in \"example_model\"\n\nInput model name to load")
model_name = input()

label_file = 'training_packed/label_names.pkl' # File path for label names file

# Create layer instances and assign loaded weights and biases
model = Newro.Model()
model.load_Model(model_name)
# Load label files

with open(label_file, 'rb') as f:
    names_result_list = pickle.load(f)
label_names = np.array(names_result_list)

def remove_duplicates(arr):
    unique_dict = {}
    unique_list = []
    
    for item in arr:
        if item not in unique_dict:
            unique_dict[item] = True
            unique_list.append(item)
    
    return unique_list
    
class_labels = remove_duplicates(label_names)

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
    output = model.forward_Pass(test_input)

    # Get the predicted class probabilities
    prediction = output[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # Print the prediction and confidence
    print("")
    print("Prediction:", class_labels[predicted_class])
    print("Confidence:", confidence)