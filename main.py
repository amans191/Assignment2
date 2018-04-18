import pandas as pd
import numpy as np
from sklearn import preprocessing, tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as mpl

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

CatFeatures = ['id', 'job', 'marital', 'education', 'default', 'housing', 'contact', 'month', 'poutcome', 'y']
CatFrame = DataFrame[CatFeatures]

ContFrame = DataFrame.drop(CatFeatures, axis=1)

# print(ContFrame.describe())
# print(CatFrame.describe())

#converting Cat features too number format
DataFrameDummy = pd.get_dummies(DataFrame, columns = CatFrame )

# print(DataFrameDummy.head())

DataFrame.to_csv('./data/trainingSetNew.csv', index=False)

#Cleaning Data + Decsion Tree

NewData = pd.read_csv('./data/trainingSetNew.csv')

target = NewData['y']

#target.to_csv('./targetfeat.csv', index=False)

numericfeat = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
numDataFrame = NewData[numericfeat]

catDataFrame = NewData.drop(numericfeat + ['id', 'y'], axis=1)

#print(catDataFrame.head())

catDataFrame.replace('?', 'NA')
catDataFrame.fillna('NA', inplace = True )

vecCatDF = pd.get_dummies(catDataFrame)

trainDataFrame = np.hstack((numDataFrame.as_matrix(), vecCatDF))

#Creating the tree thing
DecisionTreeModel = tree.DecisionTreeClassifier(criterion='entropy')

x_train, x_test, y_train, y_test = train_test_split(trainDataFrame, target, test_size=0.2, random_state=0)

trainDataFrame = np.hstack((numDataFrame.as_matrix(), vecCatDF))

DecisionTreeModel.fit(x_train, y_train)

pred = DecisionTreeModel.predict(x_test)

#test the accuracy
print('Accuracy= ' + str(accuracy_score(y_test, pred, normalize=True)))
confusionMatrix = confusion_matrix(y_test, pred)

print(confusionMatrix)


mpl.matshow(confusionMatrix)
mpl.title('Confusion matrix')
mpl.colorbar()
mpl.ylabel('True label')
mpl.xlabel('Predicted label')
mpl.show()

#
# CatDict = CatFrame.T.to_dict().values()
#
# # vectorizer = DictVectorizer( sparse= False)
# # vecCatDF = vectorizer.fit_transform(CatDict)
# #
# # encoding_dictionary = vectorizer.vocabulary_
#
#
# print(newDataFrame)



