import pandas as pd
from PIL import Image
import numpy as np
import skimage.measure
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv("./Data/english.csv")
labels = data.iloc[:, 1]
files = data.iloc[:, 0]

img_flats = []

for i in files:
    img = np.asarray(Image.open("./Data/" + i))[:, :, 0]
    small_img = skimage.measure.block_reduce(img, (100, 100), np.min)

    # flatten small array
    flat = []
    for x in small_img:
        for y in x:
            flat.append(y)
    for e in range(len(flat)):
        flat[e] = 1 if flat[e] != 0 else 0

    img_flats.append(flat)

    print(i)

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(img_flats, labels, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# training a linear SVM classifier
from sklearn.svm import SVC

svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
print('accuracy: ', accuracy)
print('svm_predictions: ', svm_predictions)