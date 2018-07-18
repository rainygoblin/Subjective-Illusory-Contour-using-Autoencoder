# Subjective Illusory Contour using Autoencoder
Predict Illusory Contour using Deep Neural Network (Keras implementation)

## Description
Illusory Contour is one of the visual illusions.
<img src="https://github.com/takyamamoto/Subjective-Illusory-Contour-using-Autoencoder/blob/master/imgs/result/AE_%E4%BA%94%E8%A7%92.png" width=100%>

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

<img src="https://github.com/takyamamoto/Subjective-Illusory-Contour-using-Autoencoder/blob/master/imgs/result/AE_%E4%BA%94%E8%A7%92.png" width=100%>
