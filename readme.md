# CSE404 Group Proejcts

- To download data, you can download from release or kaggle page.
    - `data.rar` includes the math symbols
    - `archieve.zip` includes the English

```
├── CNN.py
├── main.py
├── readme.md
├── Data/
│   ├── english.csv
│   ├── Img/
│   └── math/
│       ├── extracted_images
│       └── extracted_images-1
└── 
```

# Model Attempts
## Convolutional Neural Network
```
C:\Users\me\AppData\Local\Programs\Python\Python311\python.exe C:\Users\me\StudioProjects\CSE404Project\NewCNN.py 
                image label
0  Img/img045-049.png     i
1  Img/img052-015.png     p
2  Img/img040-031.png     d
3  Img/img017-014.png     G
4  Img/img030-032.png     T
Found 2910 validated image filenames belonging to 62 classes.
Found 495 validated image filenames belonging to 62 classes.
Found 5 validated image filenames belonging to 5 classes.
Epoch 1/10
91/91 [==============================] - 90s 973ms/step - loss: 3.6888 - accuracy: 0.1354 - val_loss: 2.5964 - val_accuracy: 0.3394
Epoch 2/10
91/91 [==============================] - 86s 949ms/step - loss: 2.0399 - accuracy: 0.4505 - val_loss: 1.8729 - val_accuracy: 0.5152
Epoch 3/10
91/91 [==============================] - 112s 1s/step - loss: 1.2252 - accuracy: 0.6615 - val_loss: 1.6432 - val_accuracy: 0.6061
Epoch 4/10
91/91 [==============================] - 114s 1s/step - loss: 0.8317 - accuracy: 0.7660 - val_loss: 1.7962 - val_accuracy: 0.5939
Epoch 5/10
91/91 [==============================] - 86s 942ms/step - loss: 0.5354 - accuracy: 0.8409 - val_loss: 1.8183 - val_accuracy: 0.5879
Epoch 6/10
91/91 [==============================] - 86s 942ms/step - loss: 0.3967 - accuracy: 0.8835 - val_loss: 1.9269 - val_accuracy: 0.6020
Epoch 7/10
91/91 [==============================] - 83s 912ms/step - loss: 0.3112 - accuracy: 0.9096 - val_loss: 1.7284 - val_accuracy: 0.6525
Epoch 8/10
91/91 [==============================] - 84s 919ms/step - loss: 0.2436 - accuracy: 0.9258 - val_loss: 1.9891 - val_accuracy: 0.6182
Epoch 9/10
91/91 [==============================] - 84s 926ms/step - loss: 0.2044 - accuracy: 0.9381 - val_loss: 1.7882 - val_accuracy: 0.6606
Epoch 10/10
91/91 [==============================] - 82s 901ms/step - loss: 0.1393 - accuracy: 0.9636 - val_loss: 2.0704 - val_accuracy: 0.6505
Prediction mapping:  {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55, 'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}
1/1 [==============================] - 0s 383ms/step
Real: i  Pred:  j
Real: p  Pred:  p
Real: d  Pred:  d
Real: G  Pred:  G
Real: T  Pred:  T
```
