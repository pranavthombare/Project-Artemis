import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers,regularizers

import sklearn.ensemble as ske
from sklearn.preprocessing import StandardScaler
from sklearn import  tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle 

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

    
data = pd.read_csv('Cleaned.csv')
X = normalize(data.drop(['Name', 'md5', 'legitimate'], axis=1)).values
y = data['legitimate'].values
print("loaded values")

print('Researching important feature based on %i total features\n' % X.shape[1])
# Feature selection using Trees Classifier
fsel = ske.ExtraTreesClassifier().fit(X, y)

model1 = SelectFromModel(fsel, prefit=True)
X_new = model1.transform(X)
print(X_new)
nb_features = X_new.shape[1]
features = []

print('%i features identified as important:' % nb_features)

xnew = [0]*nb_features
indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
for f in range(nb_features):
    xnew[f]=data.columns[2+indices[f]];
    print(" %d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))

res={}
for f in range(X.shape[1]):
    res[data.columns[f]]=0
for f in range(nb_features):
    res[xnew[f]] = 1

i=0
xd = [0]*(X.shape[1]-nb_features+1)
for f in range(X.shape[1]):
    if(res[data.columns[f]]==0):
        xd[i]=data.columns[f]
        i=i+1
    
xfinal = data.drop(xd,axis=1).values

X_train, X_test, y_train, y_test = train_test_split(xfinal, y, test_size = 0.2, random_state = 0)
print("Building model")

model = Sequential()
model.add(Dense(units=100, kernel_initializer='uniform', activation='tanh',input_shape = X_train[1].shape))
model.add(Dense(units=40, kernel_initializer='uniform', activation='tanh'))
model.add(Dense(units=60, kernel_initializer='uniform', activation='tanh'))
model.add(Dense(units=80, kernel_initializer='uniform', activation='tanh'))
model.add(Dense(units=60, kernel_initializer='uniform', activation='tanh'))
model.add(Dense(units=40, kernel_initializer='uniform', activation='tanh'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='tanh'))
model.add(Dense(1,activation='sigmoid'))

adam = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, shuffle=True)
scores = model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")

# Plot training & validation accuracy values
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Identify false and true positive rates
'''clf = algorithms[winner]
res = clf.predict(X_test)
mt = confusion_matrix(y_test, res)'''
#print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
#print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))
