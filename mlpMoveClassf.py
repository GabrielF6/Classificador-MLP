
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.datasets import fetch_openml

print(__doc__)

# Load data from https://www.openml.org/d/554
with open('../X_train.txt', 'r') as f:
    X_train = [x.replace('\r\n','') for x in f.readlines()]

with open('../y_train.txt', 'r') as f:
    y_train = f.readlines()


with open('../X_test.txt', 'r') as f:
    X_test = [x.replace('\r\n','') for x in f.readlines()]

with open('../y_test.txt', 'r') as f:
    y_test = f.readlines()



y_train = np.array(y_train)
y_test = np.array(y_test)

filtrado = []
for line in X_train:
    aux = []
    aux2 = []
    aux = line.split(" ")
    for x in range(len(aux)):
        if(aux[x] != ''):
            aux2.append(float(aux[x]))

    filtrado.append(aux2)


filtrado = np.array(filtrado)
X_train = filtrado

############################################
filtrado = []
for line in X_test:
    aux = []
    aux2 = []
    aux = line.split(" ")
    for x in range(len(aux)):
        if(aux[x] != ''):
            aux2.append(float(aux[x]))

    filtrado.append(aux2)


filtrado = np.array(filtrado)
X_test = filtrado
###########################################


for each in range(len(y_train)):
    y_train[each] = y_train[each].replace('\n', '')

print y_train
#############################################3

for each in range(len(y_test)):
    y_test[each] = y_test[each].replace('\n', '')

print y_test
##########################################33






# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
# mlp = MLPClassifier(hidden_layer_sizes=(), max_iter=180, alpha=1e-5,
#                     solver='sgd', verbose=100, tol=1e-15, random_state=1,
#                     learning_rate_init=.001)

# mlp.fit(X_train, y_train)
# print("Training set score: %f" % mlp.score(X_train, y_train))
# print("Test set score: %f" % mlp.score(X_test, y_test))


# print mlp.predict(X_test)

# sn.set(font_scale= 1.4)

from sklearn.metrics import confusion_matrix
y_true = [int(x) for x in y_test]
# y_pred = [int(x) for x in mlp.predict(X_test)]
# cm = confusion_matrix(y_true, y_pred)

# sn.heatmap(cm ,annot=True, annot_kws={"size":16})
# print cm
# plt.show()


############################################
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = [int(x) for x in clf.predict(X_test)]

cm = confusion_matrix(y_true, y_pred)

sn.heatmap(cm ,annot=True, annot_kws={"size":16})
print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))
print cm
plt.show()