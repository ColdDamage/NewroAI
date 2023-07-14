import numpy as np
import os

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
    
    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = self._backward(dvalues, y_true) / samples
    
    def _backward(self, dvalues, y_true):
        raise NotImplementedError

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[np.arange(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def _backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
        return self.dinputs
        
class Model:
    def __init__(self):
        self.layers = []
        self.activators = []
    
    def addLayer(self, weights, neurons, activation):
        layer = Layer_Dense(weights, neurons)
        self.layers.append(layer)
        self.activators.append(activation)
    
    def forward_Pass(self, inputs):
        layer_input = inputs
        i = 0
        for layer in self.layers:
            layer.forward(layer_input)
            self.activators[i].forward(layer.output)
            layer_input = self.activators[i].output
            i += 1
        return layer_input
        
    def backward_Pass(self, output, targets):
        loss_function = Loss_CategoricalCrossentropy()
        loss_function.backward(output, targets)
        dinputs = loss_function.dinputs
        for i in range(len(self.layers)):
            index = len(self.layers)-(i+1)
            self.activators[index].backward(dinputs)
            self.layers[index].backward(self.activators[index].dinputs)
            dinputs = self.layers[index].dinputs
        
        for layer in self.layers:
            layer.update_parameters(0.05)
            
    def save_Model(self, file_path):
        trained_model = {};
        i = 0
        for layer in self.layers:
            index = str(i+1)
            trained_model["dense"+index+"_weights"] = layer.weights
            trained_model["dense"+index+"_biases"] = layer.biases
            trained_model["dense"+index+"_activator"] = self.activators[i].__class__.__name__
            i += 1
        
        if(os.path.exists(file_path) == False):
            os.mkdir(file_path)
        np.savez(file_path+"/"+file_path+".npz", **trained_model)
        
    def load_Model(self, model_name):
        loaded_model = np.load(model_name+"/"+model_name+".npz", allow_pickle=True)
        print("")
        for i in range(int(len(loaded_model)/3)):
            index = str(i+1)
            if(loaded_model["dense"+index+"_activator"] == "Activation_ReLU"):
                self.addLayer(loaded_model["dense"+str(i+1)+"_weights"].shape[0], loaded_model["dense"+str(i+1)+"_biases"].shape[1], Activation_ReLU())
                print("Loaded layer "+index+": "+str(loaded_model["dense"+index+"_weights"].shape[0])+", "+str(loaded_model["dense"+index+"_biases"].shape[1])+", Activation_ReLU")
            elif(loaded_model["dense"+index+"_activator"] == "Activation_Softmax"):
                self.addLayer(loaded_model["dense"+index+"_weights"].shape[0], loaded_model["dense"+index+"_biases"].shape[1], Activation_Softmax())
                print("Loaded layer "+index+": "+str(loaded_model["dense"+index+"_weights"].shape[0])+", "+str(loaded_model["dense"+index+"_biases"].shape[1])+", Activation_Softmax")
            self.layers[i].weights = loaded_model["dense"+index+"_weights"]
            self.layers[i].biases = loaded_model["dense"+index+"_biases"]