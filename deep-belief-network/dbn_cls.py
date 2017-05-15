import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification


X_train = np.loadtxt('../data/train_dataset')
X_test = np.loadtxt('../data/test_dataset_full.txt')

Y_train = np.array([np.argmax(row) for row in np.loadtxt('../data/train_labels')])
Y_test = np.array([np.argmax(row) for row in np.loadtxt('../data/test_labels_full.txt')])

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[64, 64, 64, 10],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.05,
                                         n_epochs_rbm=20,
                                         n_iter_backprop=100,
                                         batch_size=50,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Test
Y_pred = classifier.predict(X_train)
print('Done.\nAccuracy: %f' % accuracy_score(Y_train, Y_pred))
