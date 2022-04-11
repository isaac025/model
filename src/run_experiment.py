import os
import numpy as np
import tensorflow as tf
import load_dataset.data_proc as dp
import matplotlib.pyplot as plt

array = dp.load_file_to_nparr('/home/is/Desktop/CURP/model/load_dataset/data/probe_data/probe_pressure_velocity_density_temperature_1_FOM.npy')

data = dp.dataset_loader(array)
(x,y) = np.shape(data)

train_size = int(y * 0.8)
val_size = train_size + int((y * 0.2) / 2.0)

# Labels
time_set = data[0]
time_train_labels = time_set[:train_size]
time_val_labels = time_set[train_size:val_size]
time_test_labels = time_set[val_size:]

# Data
train_data = data[1:,:train_size]
val_data = data[1:,train_size:val_size]
test_data = data[1:,val_size:]

train_data = tf.reshape(train_data, [train_size, x-1])
val_data = tf.reshape(val_data, [y-val_size-1, x-1])
test_data = tf.reshape(test_data, [y-val_size, x-1])

train_dataset = tf.data.Dataset.from_tensor_slices((time_train_labels,train_data))
val_dataset = tf.data.Dataset.from_tensor_slices((time_val_labels,val_data))
test_dataset = tf.data.Dataset.from_tensor_slices((time_test_labels,test_data))

BATCH_SIZE = 34
SHUFFLE_BUFFER_SIZE = 10

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy'])


history = model.fit(train_data, time_train_labels, epochs=10)

print("Evaluate on test data")
results = model.evaluate(val_data, time_val_labels)
print("test loss, test acc: ", results)

print("Generate predictions for 3 samples")
predictions = model.predict(test_data)
print("predictions shape:", predictions.shape)

plt.plot(time_test_labels, test_data[:,0], 'r--')
plt.xlabel('time')
plt.show()

plt.plot(time_test_labels, predictions[:,0], 'b--')
plt.xlabel('time')
plt.show()

