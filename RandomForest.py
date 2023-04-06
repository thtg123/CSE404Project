import pandas as pd
from PIL import Image
import numpy as np
import skimage.measure
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


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


classifier = RandomForestClassifier(n_estimators = 49, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
forest_predictions = classifier.predict(X_test)

# model accuracy for X_test
accuracy = classifier.score(X_test, y_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, forest_predictions)
print('accuracy: ', accuracy)
print('Forest Random_predictions: ', forest_predictions)