# Classifying-Emotion-with-Deep-Learning
## Problem Statement 
Emotion classification with deep learning has become a more contemporary topic in the machine learning age. There are many approaches to emotion classification with neural networks, but I wanted to see if I could classify emotion using both MFCCs and melspectograms; both are visualizations of audio files. 

## Data 
The data I used for this project is the Ryerson Audio-Visual Database of Emotional Speech and Song. There are 2000+ audio files in the database of actors and actresses singing and speaking two different phrases in eight different emotions. Go to https://zenodo.org/record/1188976#.XT5sat-YU5k for a free download of the dataset. 

## MFCCs 
<p align="center">
  <img width="800" height="400" src="img/mfcc_explanation.png">
</p>

<p align="center">
  <img width="800" height="400" src="img/mfcc.png">
</p>

## EDA 
<p align="center">
  <img width="600" height="400" src="img/Happy_waveplot.png">
</p>

### Image Augmentation 
#### Shifting 
<p align="center">
  <img width="600" height="400" src="img/shift_waveplot_noise.png">
</p>

#### Stretching 
<p align="center">
  <img width="600" height="400" src="img/Stretch_waveplot_noise.png">
</p>

#### Adding White Noise 
<p align="center">
  <img width="600" height="400" src="img/noise_waveplot_noise.png">
</p>

## Convolutional Neural Networks 
<p align="center">
  <img width="800" height=400" src="img/cnn.png">
</p>

## Results 
<p align="center">
  <img width="800" height=400" src="img/results.png">
</p>

