#Source: https://www.geeksforgeeks.org/working-csv-files-python/
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import csv 
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import seaborn as sn
import pandas as pd
from mpl_toolkits import mplot3d
import string
import os
from sklearn.ensemble import AdaBoostClassifier



out = []
csvout = ["max_depth, max_leaf_nodes, accuracy_score"]

x_list = list(range(2, 6, 2))
y_list = list(range(10, 30, 10))

filename = "../data/letter/letter-recognition.data"
fields = [] 
rows = []
with open(filename, 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	fields = next(csvreader)
	for row in csvreader: 
		rows.append(row) 
# newRows = []

# for r in rows:
#     newRow = []
#     for x in r:
#         newRow = newRow + [int(float(x)*10)]
#     newRows = newRows + [newRow]
#     newRow[-1] = int(newRow[-1])
# rows = newRows

splitvalue = round(len(rows) * 0.8)
trainset = rows[:splitvalue]
testset = rows[splitvalue:]

trainX = [r[1:] for r in trainset]
trainY = [r[0] for r in trainset]
testX = [r[1:] for r in testset]
testY = [r[0] for r in testset]

for max_depth in x_list:
    newOut = []
    for max_leaf_nodes in y_list:
        fileout = "MaxDepth" + str(max_depth)+"MaxLeaves" + str(max_leaf_nodes)
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
        clf = AdaBoostClassifier(base_estimator=clf, n_estimators=1000, learning_rate=1.0, algorithm='SAMME.R')
        # clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(), random_state=1, max_iter=1000)
        # clf = svm.SVC(gamma='scale')

        newTrainY = []
        for r in trainY:
            newTrainY = newTrainY + [[r]]
        clf = clf.fit((trainX), np.array(newTrainY).ravel())

        newOut += [accuracy_score(testY, clf.predict(testX))]
        csvout += [[str(max_depth), str(max_leaf_nodes), str(accuracy_score(testY, clf.predict(testX)))]]

        print(accuracy_score(testY, clf.predict(testX)))
        print(str(accuracy_score(testY, clf.predict(testX), normalize=False)) + " correct of " + str(len(testX)))
        filename = "../letter-classification/boosting/" + fileout + "/output.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(str(accuracy_score(testY, clf.predict(testX))) + "\n")
            f.write(str(accuracy_score(testY, clf.predict(testX), normalize=False)) + " correct of " + str(len(testX)))

        def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                                n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
            """
            Generate a simple plot of the test and training learning curve.

            Parameters
            ----------
            estimator : object type that implements the "fit" and "predict" methods
                An object of that type which is cloned for each validation.

            title : string
                Title for the chart.

            X : array-like, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.

            y : array-like, shape (n_samples) or (n_samples, n_features), optional
                Target relative to X for classification or regression;
                None for unsupervised learning.

            ylim : tuple, shape (ymin, ymax), optional
                Defines minimum and maximum yvalues plotted.

            cv : int, cross-validation generator or an iterable, optional
                Determines the cross-validation splitting strategy.
                Possible inputs for cv are:
                  - None, to use the default 3-fold cross-validation,
                  - integer, to specify the number of folds.
                  - :term:`CV splitter`,
                  - An iterable yielding (train, test) splits as arrays of indices.

                For integer/None inputs, if ``y`` is binary or multiclass,
                :class:`StratifiedKFold` used. If the estimator is not a classifier
                or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

                Refer :ref:`User Guide <cross_validation>` for the various
                cross-validators that can be used here.

            n_jobs : int or None, optional (default=None)
                Number of jobs to run in parallel.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
                for more details.

            train_sizes : array-like, shape (n_ticks,), dtype float or int
                Relative or absolute numbers of training examples that will be used to
                generate the learning curve. If the dtype is float, it is regarded as a
                fraction of the maximum size of the training set (that is determined
                by the selected validation method), i.e. it has to be within (0, 1].
                Otherwise it is interpreted as absolute sizes of the training sets.
                Note that for classification the number of samples usually have to
                be big enough to contain at least one sample from each class.
                (default: np.linspace(0.1, 1.0, 5))
            """
            plt.figure()
            plt.title(title)
            if ylim is not None:
                plt.ylim(*ylim)
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_sizes, train_scores, test_scores = learning_curve(
                estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")

            plt.legend(loc="best")
            return plt

        title = "Learning Curve: Letter Recognition, boosting Tree - " + fileout
        plt2 = plot_learning_curve(clf, title, trainX, trainY, (0.0, 1.1), cv=5, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5))
        # plt2.show()
        plt2.savefig("../letter-classification/boosting/" + fileout + "/learning_curve.png", bbox_inches='tight')
        plt2.close()
        # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
        df_cm = pd.DataFrame(confusion_matrix(testY, clf.predict(testX)))
        plt.figure(figsize = df_cm.shape)
        sn.set(font_scale=1.4)#for label size

        s = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})

        s.set_xlabel('Predicted labels')
        s.set_ylabel('True labels')
        s.xaxis.set_ticklabels(list(string.ascii_uppercase))
        s.yaxis.set_ticklabels(list(string.ascii_uppercase));
        # plt.show()
        plt.savefig("../letter-classification/boosting/" + fileout + "/confusion_matrix.png", bbox_inches='tight')
        plt.close()
    out += [newOut]


with open("../letter-classification/boosting/MASTER.csv", "w") as f:
    f.write("max_depth, max_leaf_nodes, accuracy_score\n")
    for row in csvout:
        f.write(str(row[0]) + ", " + str(row[1]) + ", " + str(row[2]) + "\n")

fig = plt.figure()
ax = plt.axes(projection='3d')
print(x_list)
print(y_list)
print(out)
print(np.array(out, dtype=np.float64))
ax.contour3D(np.array(x_list), np.array(y_list), np.array(out, dtype=np.float64), 50, cmap='binary')
ax.set_xlabel('max_depth')
ax.set_ylabel('max_leaf_nodes')
ax.set_zlabel('accuracy_score');
plt.savefig("../letter-classification/boosting/MASTER.png")
plt.show()