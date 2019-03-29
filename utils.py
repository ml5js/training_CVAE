import os
import tensorflow as tf
import numpy as np

# get all the files and its label
def get_files(path):
    dirs = [x[0] for x in os.walk(path)][1:]
    
    features = []
    labels = []
    count = 0
    for d in dirs:
        files = [f for f in os.listdir(d)]
        for f in files:
            data = np.load(d+'/'+f)
            for i in range(data.shape[0]):
                img = data[i].reshape(28, 28, 1)
                img = img / 255.0
                features.append(img)
                labels.append(count)
        count += 1
        if count == 1:
            break
    
    return features, labels

# create training data
def get_data(features, labels):
    train_imgs = tf.constant(features)
    train_labels = tf.constant(labels)

    training_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))

    return training_data

features, labels = get_files('quickdraw_data')
image_label_ds = get_data(features, labels)


print('image shape: ', image_label_ds.output_shapes[0])
print('label shape: ', image_label_ds.output_shapes[1])
print('types: ', image_label_ds.output_types)
print()
print(image_label_ds)
