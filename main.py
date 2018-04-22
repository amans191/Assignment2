import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

encoding = 'utf-8-sig'

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

headers = []
with open('data/dataDescription.txt', 'r') as dd:
    for line in dd:
        if line[0].isdigit():
            items = line.split(' ')
            headers.append(items[2].strip().replace(':', ''))

DataFrame = pd.read_csv('./data/trainingset.txt', names=headers, na_values=['?'])
DataFrame.head()

CatFeatures = ['job', 'loan', 'marital', 'education', 'default', 'housing', 'month', 'poutcome', 'day','duration','contact']
CatFrame = DataFrame[CatFeatures]

ContFrame = DataFrame.drop(CatFeatures + ['id', 'y'], axis=1)

CatFrame.head()

#  Clean Data
CatFrame.replace('?', 'NA')
CatFrame.replace('unknown', 'NA')
CatFrame.fillna('NA', inplace=True)

# Ready data for classifier
vecCatDF = pd.get_dummies(CatFrame)
trainDataFrame = np.hstack((ContFrame.as_matrix(), vecCatDF))

vectorizer = DictVectorizer(sparse=False)
cat_df = CatFrame.T.to_dict().values()
vec_cat_df = vectorizer.fit_transform(cat_df)
train_df = np.hstack((ContFrame.as_matrix(), vec_cat_df))

# Create decision tree (ID3)
# DecisionTreeModel = tree.DecisionTreeClassifier(criterion='entropy')
# target = DataFrame['y']
#
# x_train, x_test, y_train, y_test = train_test_split(train_df, target, test_size=0.2, random_state=0)
#
# DecisionTreeModel.fit(x_train, y_train)
#
# pred = DecisionTreeModel.predict(x_test)
# print('Accuracy= ' + str(accuracy_score(y_test, pred, normalize=True)))

# KNN
# testing KNN for different Ks using cross-validation for test error
#
# List = list(range(1, 50))
# # getting the odds ones
# neighbours = list(filter(lambda x: x % 2 != 0, List))
# CV_scores = []
#
# for k in neighbours:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
#     CV_scores.append(scores.mean())
#
# Errors = [1 - x for x in CV_scores]
#
# optimalneighbours = neighbours[Errors.index(min(Errors))]
# print("The optimal number of neighbors is %d" % optimalneighbours)
# optimal number of neighbours was found to be 41

# KNN = KNeighborsClassifier(n_neighbors=41)
# KNN.fit(x_train, y_train.values.ravel())
# pred = KNN.predict(x_test)
# print(accuracy_score(y_test, pred))
#
# KNN had better predictions, so going ahead with KNN
# training the classifier with the whole training set

KNN = KNeighborsClassifier(n_neighbors=41)
KNN.fit(train_df,DataFrame['y'])

# #readying the query data
QueryFrame = pd.read_csv('./data/queries.txt', names=headers, na_values=['?'])
QueryCatFeatures = ['job', 'loan', 'marital', 'education', 'default', 'housing', 'contact', 'month', 'poutcome']
QueryCatFrame = QueryFrame[CatFeatures]
QueryContFrame = QueryFrame.drop(CatFeatures + ['id', 'y'], axis=1)

querycat_df = QueryCatFrame.T.to_dict().values()
queryvec_cat_df = vectorizer.fit_transform(querycat_df)
query_df = np.hstack((QueryContFrame.as_matrix(), queryvec_cat_df))
query_pred = KNN.predict(query_df)

# Making asnwer file
ids = QueryFrame['id'].ravel()
withquotes = ["\"" + item + "\"" for item in query_pred]
np.savetxt('predictions.txt', np.transpose([ids, withquotes]), fmt='%.18s', delimiter=',', newline='\r\n')
#

