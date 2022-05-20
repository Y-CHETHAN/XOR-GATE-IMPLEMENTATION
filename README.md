### EX NO : 08
### DATE  : 13/05/2022 
# <p align="center">XOR GATE IMPLEMENTATION</p>
## AIM:
To implement multi layer artificial neural network using back propagation algorithm.
## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab

## RELATED THEORY CONCEPT:
Implementing logic gates using neural networks help understand the mathematical computation by which a neural network processes its inputs to arrive at a certain output. This neural network will deal with the XOR logic problem. An XOR (exclusive OR gate) is a digital logic gate that gives a true output only when both its inputs differ from each other.
<br>The truth table for an XOR gate is shown below:<br>
![image](https://user-images.githubusercontent.com/65499285/169467988-83bcb09f-85dd-41bf-91a6-885075d4f3c5.png)

## ALGORITHM:
1. Import the required libraries.
2. Create the training dataset.
3. Create the neural network model with one hidden layer.
4. Train the model with training data.
5. Now test the model with testing data.

## PROGRAM:
```python
/*
Program to implement XOR Logic Gate.
Developed by   : Y Chethan
RegisterNumber :  212220230008
*/

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

training_data=np.array([[0,0],[0,1],[1,0],[1,1]],"float32")
target_data=np.array([[0],[1],[1],[0]],"float32")

model=Sequential()
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['binary_accuracy'])
model.fit(training_data,target_data,epochs=1000)
scores=model.evaluate(training_data,target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
print(model.predict(training_data).round())

```

## OUTPUT:

![Screenshot (15)](https://user-images.githubusercontent.com/75234646/168518291-ffea8d92-0644-4301-8c38-a74c6302cd53.png)

## RESULT:
Thus the python program successully implemented XOR logic gate.
