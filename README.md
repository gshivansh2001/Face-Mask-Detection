# Face-Mask-Detection
The Project was implemented using a dataset consisting of 4120
images collected from Kaggle,Self captured images, Celebrity Face Dataset and using Bing Search API. The Project
uses a convolutional neural network to classify people with and
without masks using 3 different pre trained models used for transfer learning namely VGG-16, VGG-19 and MobileNetv2 with the last one giving the best results, having a training accuracy of 99.78% and a test
accuracy of 97.47%. The images were preprocessed, converted to arrays, normalized and data augmentation was done as there was a little data for the model to train.
The model was trained and saved using Keras, Tensorflow and implemented on real-time and static images and videos using OpenCV. The model was then deployed locally using Flask in the backend and HTML for the Frontend.
