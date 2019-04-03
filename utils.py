import os
import tensorflow as tf
import numpy as np

# get all the files and its label
def get_files(path):
    dirs = [x[0] for x in os.walk(path)][1:]
    
    features = []
    labels = []
    labels_value = []
    count = 0

    #TODO: Load Img
    for d in dirs:
        files = [f for f in os.listdir(d)] #load np
        for f in files:
            data = np.load(d+'/'+f)                
            data = data.reshape([1, 28, 28, 1])
            features.append(data)
        labels_value.append(d)
        labels.append(count)
        count += 1
    
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
