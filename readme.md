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
```
PS C:\Users\muiph\OneDrive\Documents\CSE404Project> python trainCON2.py
Shape of X [N, C, H, W]: torch.Size([32, 3, 900, 1200])
Shape of y: torch.Size([32]) torch.int64
Using cpu device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (conv_relu_stack): Sequential(
    (0): MaxPool2d(kernel_size=16, stride=16, padding=0, dilation=1, ceil_mode=False)
    (1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(512, 62, kernel_size=(3, 3), stride=(1, 1))
  )
  (linear): Linear(in_features=1860, out_features=62, bias=True)
)
Epoch 1
-------------------------------
loss: 4.129909  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.125794

Epoch 2
-------------------------------
loss: 4.125782  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.124796

Epoch 3
-------------------------------
loss: 4.121610  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.124934

Epoch 4
-------------------------------
loss: 4.117387  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.126634

Epoch 5
-------------------------------
loss: 4.115981  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.127449

Epoch 6
-------------------------------

Random Forest Evaluation:
Best performance achieved with an nEstimator value of 49, roughly sample sata size divided by 62
accuracy:  0.2785923753665689
Forest Random_predictions:  ['W' 'Q' 'M' 'C' 'l' 'e' 'W' '1' 'A' 'j' 's' 'S' 'k' '9' 'H' 'L' 'x' 'p'
 '6' 'H' 'T' 'h' 'Q' '5' 'L' 'u' 's' 'i' 'v' 'S' 'i' 'K' 'N' 'L' 'S' '6'
 'a' 'V' 'C' 'X' '3' 'M' '9' 'u' 'C' 'k' 'N' '2' 's' 'V' '6' 'O' 'I' 'j'
 '7' 'S' 'w' 'N' 'a' 'Q' 'm' 'B' '2' '2' 'k' 'O' '3' 'S' 'e' 'y' '7' 'h'
 'k' 'L' 'P' 'Y' 'C' 'j' '8' 'd' '8' 'c' 'l' 'o' 'H' 'Y' 'B' 'u' 'c' '1'
 'M' '4' 'W' 'i' 'V' 'T' 'G' 'L' 'M' 'v' 'C' 'g' '3' 'L' 'F' '2' '5' 'G'
 'i' 'l' 'G' 'L' 'b' 'g' 'E' 'r' 'h' 'S' 'n' 'u' 'z' 't' 'N' 'l' 'b' '6'
 'n' '5' 'x' 'L' 'o' 's' 'V' 'F' 'x' 'v' 'H' 's' 'L' 'w' 'O' 'r' 'q' 'N'
 'g' 'P' 'E' 'T' 'y' 'p' 'k' 'G' 'q' 'T' 'k' 'Y' 'x' 'R' 'L' 'N' 'h' 'w'
 'E' '6' '0' 'l' 'd' 'v' '4' 'E' '4' 't' 'L' 'w' 'b' 'q' 'G' 'P' 'Q' 'B'
 'N' '3' 'H' 'w' 'e' 'k' 'n' 'y' '2' 'N' 'Q' 'c' 'K' 'M' 'D' 'w' 'V' 'i'
 'R' 'J' 'V' 'Z' 'j' '6' 'm' 'J' 'L' '4' 'd' 'C' '6' 'R' 'o' '0' 'E' 'P'
 'O' '9' 'h' 'p' 'e' 'e' 'h' 'C' 't' 'K' '3' 'H' 'F' 'G' 'r' 'M' 'V' 'R'
 's' 'A' 'v' 'n' 'J' 'E' 'r' 'u' '8' 'L' 'O' 'X' 'x' 'V' '4' 'x' '0' 'o'
 '6' 'r' 'V' '8' 'y' 'u' 'B' 'o' 'R' 'R' '1' 'y' '2' 'G' 'I' 'b' 'd' 'Z'
 'q' 'E' 'p' 'O' 'C' 'c' 'i' 'w' 'I' 'H' 'D' '1' 'C' 'n' 'L' 't' '4' '8'
 'w' 'E' '4' '8' 'S' 'x' 'S' 's' 'H' 'U' 'r' 'e' 'y' 'n' 'd' 's' '4' 'I'
 'v' 'w' '3' '3' 's' 'm' '1' 'e' 't' 'p' '2' 'P' 'l' 'b' 'M' 'r' 'D' 'U'
 'h' 'p' 'n' 'F' 'v' 'J' '7' 'O' 'P' 'r' 'H' 'a' 'I' 'v' 'X' '3' '0' 'V'
 'X' 'q' 'D' '6' 'R' 'F' 'n' 'j' 'V' '9' '0' 'W' 'r' 'k' 'q' 'w' 'Q' 'U'
 '9' 'P' 'P' 'T' 'F' 'C' 'k' 'r' '2' 'i' 'T' 'u' 'H' 'r' 'm' 'O' 'K' 'j'
 '2' 'G' '8' 'p' 'j' '1' 'i' 'd' 'y' 'h' 'A' 't' 'V' 'E' 'O' 'd' 'P' 'T'
 'k' 'p' 'h' 's' 'F' 'l' 'Z' 'i' 'S' 'A' '3' 'G' 'I' 'w' 'Y' 'n' 'E' 'b'
 'K' '2' 'l' '7' 'F' 'V' '8' 'o' 'Y' 'K' 's' 'C' '0' 'R' 'T' 'b' 'f' '5'
 'h' 'b' 'K' 'e' 'N' 'V' '6' 'T' 'q' '4' 'U' '7' 's' 'U' 'e' 'v' '2' '6'
 'N' '1' 'M' 'f' 'r' 'B' 'q' 'D' 'f' '8' 'H' 'W' 'j' 'F' 'B' 'z' 't' 'n'
 'F' '9' 's' 'E' 'j' 'g' 'k' 'c' 'P' 'w' 'r' 'h' 'u' 'm' 'A' 'y' 'W' 'o'
 'S' '8' 'k' 'P' 'A' 'Z' '9' 'a' 'q' 'R' 'E' 'M' 'w' 'd' 'Y' '4' 'T' 'L'
 'S' '8' 'M' 'k' '7' 'S' 'I' 'O' 'x' '8' 'c' '5' 'w' '7' '3' 'p' 'T' 'd'
 'w' '9' 'K' 'O' 'n' '5' 'D' 'Q' 'x' '1' 'K' 'u' 'Q' '7' 'I' 'V' 'P' '1'
 'W' 'N' 'N' 'j' '8' 'K' '4' 'Z' 'F' 'X' 'E' 'd' 'a' 'T' '6' 'v' 'u' '4'
 'L' '3' '2' 'X' 'G' 'a' 'N' 'o' 'a' 'Y' 'M' 'u' 'e' '4' 'M' 's' 'N' 'R'
 'j' 'T' 'M' 'N' 'O' 'B' 'R' 'O' 'f' 'j' 'g' 'j' 'A' 'e' 'E' 'o' '1' '2'
 '0' 'y' 'U' 'S' 'P' 'b' 'V' 'V' 'N' 'x' 'R' 'Q' 'd' 'H' 's' 'i' 'V' 'c'
 'T' 'N' 'i' 'A' 'e' 'L' 'F' 'o' 'g' 'B' 'Y' 'b' 'L' 'z' 'a' 'M' 'V' '1'
 'V' 'Z' 'k' 'u' 's' 'i' 'H' 'y' 'Y' 'a' 'u' 'S' 'x' 'L' 'y' 'n' 'B' 'c'
 '2' 'r' 'a' 'W' 'Y' 'y' 'u' 'j' '0' '2' 'b' 'A' 'Q' 'd' 'B' 'Q' '6' 'W'
 'B' 'e' '7' '2' 'm' 'f' 'v' 'V' 'C' 'w' 'x' 'q' 'M' '6' 'e' 'u']
loss: 4.116341  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.127278

Epoch 7

Epoch 10
-------------------------------
loss: 4.113594  [    0/ 2728]
Test Error: 
 Accuracy: 2.6%, Avg loss: 4.121228 

Done!
PS C:\Users\muiph\OneDrive\Documents\CSE404Project> 
```
