import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time
import cpuinfo
import platform
import GPUtil

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

start = time.time()

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64))

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_images, 
    train_labels, 
    epochs=10, 
    validation_data=(test_images, test_labels)
)

t = time.time()-start
t = t/10

print('ave. time for each epoch %.5f' % t)

f=open('benchmark_tensorflow.csv','a')

sysver = platform.python_compiler()
gpus = GPUtil.getGPUs()
gpu0 = gpus[0]
perf = "%s, %s, %1.3fs\n" % (gpu0.name, sysver, t)
print(perf)
f.write(perf)
f.close()