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

<br><br><br><br><br><br>
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
<br><br><br>
## OUTPUT:
![image](https://user-images.githubusercontent.com/65499285/169468591-fe5976a0-8c59-4909-959e-96a1724cda98.png)

![image](https://user-images.githubusercontent.com/65499285/169468736-356002c7-db59-41d3-8857-0c326995e75e.png)

![image](https://user-images.githubusercontent.com/65499285/169468521-7d4a40c8-a621-4e27-8ede-b476025e01fc.png)

## RESULT:
Thus the python program successully implemented XOR logic gate.
