import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification

def read_setting(setting_file):
    tmp = {}
    for line in setting_file:
        tmp[line.split()[0]] = line.split()[1]
    tmp['hidden_layers_structure'] = [int(num) for num in tmp['hidden_layers_structure'].split(',')]
    return tmp

setting = read_setting(open('setting.txt'))
file_out = open('../result/dbn.log', 'a')


X_train = np.loadtxt('../data/train_dataset_new.txt')
X_test = np.loadtxt('../data/test_dataset_new.txt')

Y_train = np.array([np.argmax(row) for row in np.loadtxt('../data/train_labels_new.txt')])
Y_test = np.array([np.argmax(row) for row in np.loadtxt('../data/test_labels_new.txt')])

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=setting['hidden_layers_structure'],
                                         learning_rate_rbm=float(setting['learning_rate_rbm']),
                                         learning_rate=float(setting['learning_rate']),
                                         n_epochs_rbm=int(setting['n_epochs_rbm']),
                                         n_iter_backprop=int(setting['n_iter_backprop']),
                                         batch_size=int(setting['batch_size']),
                                         activation_function=setting['activation_function'],
                                         dropout_p=float(setting['dropout_p']),
                                         l2_regularization=float(setting['l2_regularization']),
                                         contrastive_divergence_iter=int(setting['contrastive_divergence_iter']))

classifier.fit(X_train, Y_train)

# Test
Y_pred = classifier.predict(X_train)
accuracy = accuracy_score(Y_train, Y_pred)
print('Done.\nAccuracy: %f' % accuracy)

file_out.write('\n\n-------------------------------\n\n')

for line in open('setting.txt'):
    file_out.write(line)

file_out.write(str(accuracy) + '\n')
file_out.close()




