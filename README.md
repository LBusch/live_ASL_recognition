# Lightweight Real-Time ASL Alphabet Recognition
PyTorch project for real-time recognition of the ASL (American Sign Language) alphabet using Machine Learning. MobileNetv2. ASL alphabet signs can be recognized in real time using a webcam.

With this project a MobileNetv2 can be trained for ASL alphabet recognition on the ASL Alphabet dataset [^1] from kaggle. The dataset consists of 87,000 images of ASL hand signs. It contains 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING. The dataset does not come with a test set to encourage the use of real-world test images. This project includes a script for self-recording ASL data from a webcam for fine-tuning and testing. The project also includes a script for using a trained model for real-time ASL alphabet recognition on live image data from a webcam. 

Using pre-trained weights from image classification on the ImageNet dataset, the MobileNetv2 was trained on the ASL Alphabet dataset for 10 epochs. On self-captured test data the model achieved an accuracy of 67.8%. Then, the model was fine-tuned on a separate set of self-captured training data for 5 epochs. After fine-tuning, the model achieved an accuracy of 98,3% on the test data.

# Plots
<img width="1200" height="500" alt="asl_mobilenetv2_kaggle" src="https://github.com/user-attachments/assets/f573fe21-71a3-4fc0-be39-616937f0e9b7" />


# How to Use
`download_kaggle_dataset.py`: Downloads the ASL Alphabet dataset from kaggle. Requires a properly set up kaggle API key. Alternatively, download the dataset manually from kaggle.

`capture_image_data.py`: Used for self-capturing ASL alphabet image data for fine-tuning or testing. You can capture images for all signs after another or for only one specified sign.

`train_model.py`: Trains and saves a MobileNetv2 on a specified dataset for ASL alphabet recognition. Optionally a saved model can be selected as a starting point for fine-tuning.

`test_model.py`: Tests a model on a specified dataset and outputs metrics like a confusion matrix and the model's accuracy.

`live_inference.py`: Performs real-time inference of a saved model on live image data from a webcam.

`image_inference.py`: Performs inference of a saved model on images in a specified directory. Predictions for each image are saved in a CSV file.

# Video Demo of Live ASL Alphabet Recognition
https://github.com/user-attachments/assets/b366f8eb-1663-4bf7-a6f0-2a374fb207fc

# Video Demo of Image Data Capturing for Training/Testing
https://github.com/user-attachments/assets/1fcf6eb0-817c-4538-a7e4-fdd448644531

## References
[^1]: Nagaraj, Akash. (2018). *ASL Alphabet* [Data set]. Kaggle. https://www.kaggle.com/dsv/29550  
DOI: [10.34740/KAGGLE/DSV/29550](https://doi.org/10.34740/KAGGLE/DSV/29550)




