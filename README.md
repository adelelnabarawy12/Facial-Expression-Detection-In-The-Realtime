# Facial-Expression-Detection-In-The-Realtime
- The human's facial expressions is very important to detect thier emotions and sentiment. It can be very efficient to use to make our computers make interviews. Furthermore, we have robots now can detect the human's emotions and based on thats take an action .etc. So, It will be better to provide a tool or model for this.

- This project meant with detecting the 7 different human's facial expressions (Natural - Happy - Fear - Sad - Surprise - angry).

- <b><i>The libraries used:</i><b/> Tensorflow, Keras, OpenCV, Skimage, Numpy, Seaborn, Matplotlib and Sklearn (for model evaluation).  

- <b><i>The Dataset used:</i><b/> https://www.kaggle.com/msambare/fer2013

<img src = '/imgs/training_data_dist.png' width = '400px'/>

- <b><i>A test sample in the realtime:</i><b/>

<img src = '/imgs/test.png' width = '400px'/>

## The Preprocessing:
  - <b><i>Oversampling and Undersampling techniques:</i><b/>
  
     By making an undersampling for the majority class which is "happy" and oversampling by repeating another samples and data augmentation.
     
  <img src = '/imgs/balanced_dist.png' width = '400px'/>
    
  - <b><i>Data Augmentation</i><b/>
  
  <img src = '/imgs/data_augmentation.png' width = '400px'/>
  
## The model architecture used (CNN + LSTM + FC):
  
  <img src = '/imgs/dev_model_arch.png' width = '400px'>
  
## The model evaluation:
  - <b><i>Model Accuracy:</i><b/> <b>60%</b>
  - <b><i>The confusion matrix:</i><b/>
    
  <img src = '/imgs/confusion_matrix.png' width = '400px'>
  
  - <b><i>The Model training and validation performance:</i><b/>
    
  <img src = '/imgs/model_training_performance.png'>
  
