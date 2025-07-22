# Lightweight Real-Time ASL Alphabet Recognition
PyTorch project for real-time recognition of the ASL (American Sign Language) alphabet using MobileNetv2. ASL alphabet signs can be recognized in real time using a webcam.

In this project a MobileNetv2 is trained for ASL alphabet recognition on the ASL Alphabet dataset [^1] from kaggle. Then, the model is fine-tuned and validated on self-captured train and test data. Finally, the model can be used for real-time ASL alphabet recognition on live data from a webcam.

# How to Use
`download_kaggle_dataset.py`: Downloads the ASL Alphabet dataset from kaggle. Requires a properly set up kaggle API key. Alternatively, download the dataset manually from kaggle.

`train_model.py`: Trains and saves a MobileNetv2 on a specified dataset for ASL alphabet recognition.

`test_model.py`: Tests a pretrained model on a specified dataset.

`capture_image_data.py`: Used for self-capturing ASL alphabet image data for fine-tuning or testing. You can capture images for all signs after another or for only one specified sign.

`live_inference.py`: Performs real-time inference of a saved model on live data from a webcam.

`image_inference.py`: Performs inference of a saved model on images in a specified directory.

# Video Demo
https://github.com/user-attachments/assets/b366f8eb-1663-4bf7-a6f0-2a374fb207fc

## References
[^1]: Nagaraj, Akash. (2018). *ASL Alphabet* [Data set]. Kaggle. https://www.kaggle.com/dsv/29550  
DOI: [10.34740/KAGGLE/DSV/29550](https://doi.org/10.34740/KAGGLE/DSV/29550)
