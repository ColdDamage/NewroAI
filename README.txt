After writing the readme below, I trained a new model in order to get a smaller filesize for the trained model. Github wouldn't let me upload the original trained model of 187600KB. The new model was trained with 64 neurons the hidden layer and reached a loss of 0.00 at 37 epochs and the training was done over a total of 50 epochs. The new model is considerably faster and seems to be just as accurate and confident as the previous one, from the short testing I did.

----------------------------------------

I made this project to learn how to create a neural network and understand how they work. I set a goal to have the model differentiate between images of human faces and images of flowers. The model flower_network_model.npz has been trained with a total of 400 images over 100 epochs with batch sizes of 8, but good loss values were achieved around 50-60 epochs.

I probably overshot how large the model needs to be, similar results and probably better performance could be achieved with a smaller model.

Here are short summaries of the different files:

-------------training.py---------------
Use this to create and train a new model.

To train a new model, define the training data folder, model name and layers. If you have more or fewer layers, you also need to adjust the forward and backward pass.

You will need to adjust the labling for the images unless you have exactly 200 images of two different categories and they are sorted so that the first category's images are in the folder before the other category.

It's way easier to just put the training images in different folders if you have more than two categories, then have the lables match the folder names.

To make saving the model possible, you need to also define the trained_model dict according to your architechture.

-------------retraining.py---------------
This file is for further training of an existing model. Define the model name you want to train (note that it will be overwritten, make a backup) and adjust the layers to match the architechture of your model. If you have more or fewer layers, you also need to adjust the forward and backward pass.

You also have to define the trained_model dict according to your architechture.

-------------flower_loader.py---------------
This is a file that loads flower_network_model.npz for demoing the model. If you want to make a new model and use it in a similar way, you need to change the model_name and make the layer architechture match your model. If you have more or fewer layers, you also need to adjust the forward pass.

If you have more or fewer than 2 outputs, you also need to adjust the printing at the end of the loop.

-------------Newro.py---------------
This file just contains classes needed to create a neural network with this model. You probably don't need to touch these even if you make a new model using it.