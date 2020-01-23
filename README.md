# Classification and Detection with Convolutional Neural Networks

## Required Software
- Python 3.6
- Tensorflow
- Keras
- numpy
- scipy
- pandas
- opencv 3.0 (required for Python 3+)

## Needed Files
- digit_cnn.py
- MyCallBack.py
- plot_chart.py
- plot_it.py
- process_video.py
- proj_pipeline.py
- project_utilities.py
- run_cnn.py
- vgg16_model.h5
- *.pickle files if you want to test a quick training routine

## Alternative Weight file Location
- Model & Weights: https://drive.google.com/open?id=1QBoNr6Rnqvzgtx_-ZtF3uvfSZgYcX9dZ

## How to run
Simply run the run_cnn.py file and it will automatically create the 5 output images (1.png through 5.png) by processing image1.png through image5.png and placing them in the graded_images directory.  The weights and model data is automatically imported from vgg16_model.h5.

If you want to run a short iteration of the training procedure, you can set the variable "train_it" to True and it will pull in data from the .pickle files.  It will output a few file artifacts which can be ignored/removed if desired.  The only purpose of including these were to verify the training worked correctly.

The CNN model architecture is stored in the digit_cnn.py file and the majority of the pipeline steps are in the proj_pipeline.py file.  

## Video(s)
- Example video can be found here:
  * https://www.youtube.com/watch?v=xWz1FvkK7Kc

