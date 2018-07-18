# Subjective Illusory Contour using Autoencoder
Predict Illusory Contour using Deep Neural Network (Keras implementation)

## Description
[Illusory Contour](https://en.wikipedia.org/wiki/Illusory_contours) is one of the visual illusions.
<img src="https://github.com/takyamamoto/Subjective-Illusory-Contour-using-Autoencoder/blob/master/imgs/results/testset_hexagon.png" width=100%>

### model
Model is Autoencoder model
<img src="https://github.com/takyamamoto/Subjective-Illusory-Contour-using-Autoencoder/blob/master/imgs/model.png" width=50%>

## Requirement
```
Python == 3.6
Keras >= 2.0
Tensorflow >= 1.8
OpenCV >= 3.4
tqdm >= 4.23
```
## Usage
1. Run `generate_data.py`
1. Run `train_autoencoder.py`

## Result
### Learning Curve
<img src="https://github.com/takyamamoto/Subjective-Illusory-Contour-using-Autoencoder/blob/master/imgs/LearningCurve.png" width=50%>  

### Output  
Test set result  
- Input
- Ground truth
- Predicted
- |Predicted - Input|

<img src="https://github.com/takyamamoto/Subjective-Illusory-Contour-using-Autoencoder/blob/master/imgs/results/testset_hexagon.png" width=100%>
