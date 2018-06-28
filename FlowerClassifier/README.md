# Iris flower dataset classifier
Dataset used: http://archive.ics.uci.edu/ml/datasets/Iris

### Using only Python (iris_flower_classifier.py)
- After comparing different models, the SVM had the highest predicted accuracy.
- The `make_predictions` function uses this model to produce predictions concerning which flower belongs to which species.
- Used a more OOD approach than the tutorial to encapsulate components.

Source for this tutorial: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

### Using Keras (keras_iris_classifier.py)
- Task: multi-class classification problem (3 classes)
- The NN was defined, followed by its evaluation using scikit-learn with k-fold cross validation

Source for this tutorial: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/