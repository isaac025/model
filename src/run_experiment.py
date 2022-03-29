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

train_dataset = tf.data.Dataset.from_tensor_slices((train_data,time_train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_data,time_val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data,time_test_labels))

BATCH_SIZE = 34
SHUFFLE_BUFFER_SIZE = 10

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# Model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy'])


history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

print("Evaluate on test data")
results = model.evaluate(test_dataset)
print("test loss, test acc: ", results)

print("Generate predictions for 3 samples")
predictions = model.predict(val_data)
print("predictions shape:", predictions.shape)

plt.plot(val_data[:,0], time_val_labels, 'r--')
plt.ylabel('time')
plt.show()

plt.plot(predictions[:,0], time_val_labels, 'r--')
plt.ylabel('time')
plt.show()


