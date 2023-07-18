I made this project to learn how to create a neural network and understand how they work. I set a goal to have the model differentiate between images of human faces and images of flowers. The example model has been trained for this purpose with the settings in the model_trainer.py

Here are short summaries of the different files:

-------------training_data_packer.py----------
I found it easiest to pack the training data into arrays before starting training. This script does that when you organize your training images into folders that are named per category. The script produces an array of the images in each subfolder and another array for lables, as well as lable names. To use this, just set the directory path on line 44 to match the folder where all of your subfolders are.

-------------model_trainer.py-----------------
Use this to create and train a new model.

To train a new model, define the training variables to match your training data and settings and input the desired layers and neurons.

-------------model_loader.py------------------
This is a file that loads a trained model for testing. When you run the script, it will ask you for the model name and image filename from the test_images folder.

-------------Newro.py-------------------------
This file just contains classes needed to create a neural network with this model. You probably don't need to touch these even if you make a new model using it.