from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import cross_validation, grid_search
from multilayer_perceptron import MultilayerPerceptronClassifier
from multilayer_perceptron_sparse import MultilayerPerceptronClassifierSparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC, SVC
from numpy import random

# open file
# set fname to reo.txt for ReoQueries and geo.txt for GeoQueries
fname = 'reo.txt'
X = []
y = []

# read file into array
with open(fname) as f:
    for line in f:
        record = line.split(',')

        X.append(record[0] + ' ' + record[1])
        y.append(int(record[-1]))

# create vectorizer
transformer = CountVectorizer(stop_words="english", binary=True)
X = transformer.fit_transform(X)
y = np.array(y)

# create classifier list
clfs = [('Logistic Regression ', LogisticRegression(penalty='l1')),
        ('SVM Linear ', SVC(kernel='linear', probability=True)), ('SVM Poly ', 
                        SVC(kernel='poly', probability=True)), ('SVM RBF ', 
                        SVC(kernel='rbf', probability=True)),
        ('MLP ', MultilayerPerceptronClassifier(n_hidden=25)),
        ('Sparse MLP ', MultilayerPerceptronClassifierSparse(n_hidden=25, 
                                                             sparsity_param=0.12))]

print "Wait for the magic ..."

# apply cross validation
for (name, clf) in clfs:
    random.seed(0)
    score = cross_validation.cross_val_score(clf, X, y, cv=3, 
                                             scoring='roc_auc')

    print name + 'AUC: ', np.around(np.mean(score), 2), 'STD :', \
                          np.around(np.std(score), 3)
