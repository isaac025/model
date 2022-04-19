import os
import numpy as np
import tensorflow as tf
import load_dataset.data_proc as dp
import matplotlib.pyplot as plt

array = dp.load_file_to_nparr('/home/is/Desktop/CURP/model/load_dataset/data/probe_data/probe_pressure_velocity_density_temperature_1_FOM.npy')
array2 = dp.load_file_to_nparr('/home/is/Desktop/CURP/model/load_dataset/data/probe_data/probe_pressure_velocity_density_temperature_2_FOM.npy')

data = dp.dataset_loader(array)
data2 = dp.dataset_loader(array2)

ls = list()
for i in data:
    ls.append(i)

ls1 = list()
for i in data2[:1][0]:
    ls1.append(1)

ls2 = list()
for i in data2:
    ls2.append(i)

y_train = data2[:1][0] 
x_test = np.array(list(zip(*ls2)))
x_test = np.expand_dims(x_test,axis=1)
x_train = np.array(list(zip(*ls)))
x_train = np.expand_dims(x_train,axis=1)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1,5)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

model.summary()

predictions = model(x_train[:1]).numpy()
print(predictions)

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_train, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
print(probability_model(x_test[:5]))

