# Project Title

SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images.

There are 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data
Comes in two formats:
1. Original images with character level bounding boxes.
2. MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors at the sides).

My project is to train a CNN to identify numbers upto a limit of three digits.

## Getting Started

1. Go to http://ufldl.stanford.edu/housenumbers/ and download the training, test and validation datasets and transfer it into a image_dir folder.
2. Transfer the code and and image_dir folder into a single folder
3. change the file paths accordingly in gsv_whole_num_process.py and gsv_matlab.py
4. Run the files in the following order -gsv_matlab.py,gsv_whole_num_process.py,model1.py


### Prerequisites
1. h5py
2. numpy
3. tensorflow
4. os
5. pickle
6. matplotlib.pyplot
7. PIL

## Deployment

While training the model (while the model1.py is running), u can run the command tensorboard --logdir /tmp/Mark2_logs

## Acknowledgments

The model was highly influenced by the RyannG's blog.
You can read about it here - https://github.com/RyannnG/Capstone-Google-SVHN-Digits-Recognition


