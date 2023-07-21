# MarkUs Exam Matcher Model Training

## General Information
This repository contains the code that trained the CNNs used in the MarkUs
automatic exam matching pipeline.

## Dataset Information
Our numeric CNN is trained on the version of the MNIST dataset that is 
downloaded from PyTorch version 2.0.0. As per the PyTorch documentation, this
dataset can be found [here](https://yann.lecun.com/exdb/mnist/).

## Training Process (Numeric)
The model was trained for 70 epochs, with the goal of training until validation
accuracy began to decrease. To see the training loss and validation accuracy
as training progressed, one can write the following in `main.py`:
```python
if __name__  == '__main__':
    results_dict = load_training_statistics(f'{config["RELATIVE_STATISTICS_LOC"]}/statistics.pkl')
    plot_results(results_dict)
```
At around ~20 epochs, the validation accuracy began to plateau at around 99%,
but it never began to decrease. While this was occurring, the training loss
(stochastically) hovered within distance 0.001 from 0. This indicates that
the model had reached its maximum potential under the tuned hyperparameters
(which can be viewed in `config.py`). Any further training after the model
has begun to plateau will cause the model to overfit the training data. Even
if the validation accuracy is not decreasing, it is advisable to perform early
stopping here, since the extra training can only cause overfitting without
generalization improvement if the validation accuracy is plateauing.
Consequently, the model chosen was the model trained up until the end of
epoch 24, as this is when the training loss was approximately within 0.001 of
0 and the model had achieved 99% validation accuracy.

