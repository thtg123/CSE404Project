import pandas
import random
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

data_path = r"./Data/"

data = pandas.read_csv(data_path + '/english.csv')
rand = random.sample(range(len(data)), 500)
val = pandas.DataFrame(data.iloc[rand, :].values, columns=['image', 'label'])
# remove the added data
data.drop(rand, inplace=True)

rand = random.sample(range(len(val)), 5)
test = pandas.DataFrame(val.iloc[rand, :].values, columns=['image', 'label'])
# remove the added data
val.drop(rand, inplace=True)

print(test)

train_data_generator = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2)
data_generator = ImageDataGenerator(rescale=1/255)
training_data = train_data_generator.flow_from_dataframe(
    dataframe=data,
    directory=data_path,
    x_col='image',
    y_col='label',
    target_size=(100, 100),
    class_mode='categorical'
)
validation_data_frame = data_generator.flow_from_dataframe(
    dataframe=val,
    directory=data_path,
    x_col='image',
    y_col='label',
    target_size=(100, 100),
    class_mode='categorical'
)
test_data = data_generator.flow_from_dataframe(
    dataframe=test,
    directory=data_path,
    x_col='image',
    y_col='label',
    target_size=(100, 100),
    class_mode='categorical',
    shuffle=False
)

model = tf.keras.models.Sequential()
model.add(Conv2D(filters=30, kernel_size=3, activation='relu', input_shape=[100, 100, 3]))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Conv2D(filters=30, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(units=600, activation='relu'))
model.add(Dense(units=62, activation='softmax'))

# compile cnn
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=training_data, validation_data=validation_data_frame, epochs=10)

print("Prediction mapping: ", training_data.class_indices)
pred = model.predict(test_data)

# switcher shows our network mapping to the prediction
switcher = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "A",
            11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K",
            21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U",
            31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "a", 37: "b", 38: "c", 39: "d", 40: "e",
            41: "f", 42: "g", 43: "h", 44: "i", 45: "j", 46: "k", 47: "l", 48: "m", 49: "n", 50: "o",
            51: "p", 52: "q", 53: "r", 54: "s", 55: "t", 56: "u", 57: "v", 58: "w", 59: "x", 60: "y",
            61: "z"}

maxes = list(pandas.DataFrame(pred).idxmax(axis=1))
for i in range(len(test)):
    print("Real:", test.at[i, 'label'], " Pred: ", switcher.get(maxes[i], "error"))
