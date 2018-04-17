import pandas as pd
from sklearn import preprocessing, tree
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

encoding = 'utf-8-sig'

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

# headersfile = open('./data/headers.txt')
# headers = headersfile.read().split()

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

CatFrame.replace('?', 'NA')
CatFrame.fillna('NA', inplace = True )


CatDict = CatFrame.T.to_dict().values()

# vectorizer = DictVectorizer( sparse= False)
# vecCatDF = vectorizer.fit_transform(CatDict)
#
# encoding_dictionary = vectorizer.vocabulary_


vecCatDF = pd.get_dummies(CatFrame)

newDataFrame = (( ContFrame.as_matrix(), vecCatDF ))

target = newDataFrame['y']

print(newDataFrame)

#Creating the tree thing
DecisionTreeModel = tree.DecisionTreeClassifier(criterion='entropy')

x_train, x_test, y_train, y_test = cross_validation.train_test_split(newDataFrame, target,
                                                                     test_size=0.2, random_state=0)

DecisionTreeModel.fit(x_train, y_train)

predictions = DecisionTreeModel.predict(x_test)

#test the accuracy
print('Accuracy= ' + str(accuracy_score(y_test, predictions, normalize=True)))
confusionMatrix = confusion_matrix(y_test, predictions)
print(confusionMatrix)


