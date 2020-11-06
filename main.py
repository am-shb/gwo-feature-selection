import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from GwoOptimizer import GWO

num_features = 20

def fitness_function(positions):
    features = np.where(positions>=0.4999)[0]
    #print('selected_features:', features)
    
    #print(train_df.head())

    train_xf = train_x.iloc[:, features]
    test_xf = test_x.iloc[:, features]

    knn_classifier = KNeighborsClassifier(n_neighbors=7)
    #knn_classifier = svm.SVC()
    knn_classifier.fit(train_xf, train_y)
    
    accuracy = knn_classifier.score(test_xf, test_y)
    
    #print('Accuracy:', accuracy)

    w = 0.9
    
    return -(w*accuracy + (1-w) * 1/(len(features)))

# Load the data
data_df = pd.read_csv('mobile_price_dataset.csv')
train_data, test_data = train_test_split(data_df)
# onehot encoding
#enc = OneHotEncoder(handle_unknown='ignore')
#train_df = enc.fit_transform(train_df.to_numpy())
#test_df = enc.fit_transform(test_df)

train_x, test_x, train_y, test_y = train_data.iloc[:, :num_features], test_data.iloc[:, :num_features], train_data.iloc[:, -1], test_data.iloc[:, -1]
print ('train_x shape:', train_x.shape)
print ('train_y shape:', train_y.shape)
print ('test_x shape:', test_x.shape)
print ('test_y shape:', test_y.shape)

#train_y = train_df.iloc[:, 41]
#test_y = test_df.iloc[:, 41]


# Feature selection using GWO

fit = GWO(fitness_function, 0, 1, num_features, 10, 20)
selected_features = np.where(fit>0.5)[0]
#selected_features = [9,13,14,39]
print(selected_features)

train_x = train_x.iloc[:, selected_features]
test_x = test_x.iloc[:, selected_features]

knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(train_x, train_y)


predicted = knn_classifier.predict(test_x)
print(confusion_matrix(test_y, predicted))
print(accuracy_score(test_y, predicted))
