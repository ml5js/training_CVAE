import os
import tensorflow as tf
import numpy as np
import keras
import random

# # get all the files and its label
# def get_files(path, args):
#     dirs = [x[0] for x in os.walk(path)][1:]
    
#     training_features = None
#     validation_features = None
#     training_labels = None
#     validation_labels = None
#     labels_value = []
#     count = 1
#     imgs = ['.png', '.jpg', '.jpeg']

#     # Load data from directory
#     for d in dirs:
#         files = [f for f in os.listdir(d)] #load np
#         for f in files:
#             # Load img
#             if any(st in f for st in imgs):
#                 img_raw = tf.io.read_file(d+'/'+f)
#                 img_tensor = tf.image.decode_image(img_raw)
#                 img_tensor = tf.image.resize(img_tensor, [1, args.image_size, args.image_size, args.image_depth])
#                 if random.randint(1, 100) % 10 != 0:
#                     if training_features is None:
#                         training_features = np.copy(img_tensor)
#                         training_labels = count * np.ones((1, 1))
#                     else:
#                         training_features = np.concatenate([training_features, img_tensor], axis=0)
#                         new_labels = count * np.ones((1,1))
#                         training_labels = np.concatenate([training_labels, new_labels], axis = 0)
#                 else:
#                     if validation_features is None:
#                         validation_features = np.copy(img_tensor)
#                         validation_labels = count * np.ones((1,1))
#                     else:
#                         validation_features = np.concatenate([validation_features, img_tensor], axis=0)
#                         new_labels = count * np.ones((1,1))
#                         validation_labels = np.concatenate([validation_labels, new_labels], axis=0)

#             # Load npy file
#             else:
#                 data = np.load(d+'/'+f)                
#                 data = data.reshape([-1, args.image_size, args.image_size, args.image_depth])
#                 length = data.shape[0]
#                 # concatenate arrays to get the training data and labels
#                 if training_features is None:
#                     training_features = np.copy(data[:length-(length//10),:,:,:])
#                     training_labels = count * np.ones((length-(length//10),1))
#                 else:
#                     training_features = np.concatenate([training_features, data[:length-(length//10),:,:,:]], axis=0)
#                     new_labels = count * np.ones((length-(length//10),1))
#                     training_labels = np.concatenate([training_labels, new_labels], axis=0)

#                 # concatenate arrays to get the validation data and labels
#                 if validation_features is None:
#                     validation_features = np.copy(data[length-(length//10):,:,:,:])
#                     validation_labels = count * np.ones(((length//10),1))
#                 else:
#                     validation_features = np.concatenate((validation_features, data[length-(length//10):,:,:,:]), axis=0)
#                     new_labels = count * np.ones(((length//10),1))
#                     validation_labels = np.concatenate([validation_labels, new_labels], axis=0)
#                 data = None
        
#         labels_value.append(d.split("\\")[-1])
#         count += 1
    
#     # TODO: Fix memory leak
#     # Pad the data to fit the batch size
#     # For features
#     training_residual_shape = training_features.shape[0] % args.batch_size
#     validation_residual_shape = validation_features.shape[0] % args.batch_size
#     if training_residual_shape != 0:
#         padding = np.zeros((args.batch_size - training_residual_shape, args.image_size, args.image_size, args.image_depth))
#         new_features = np.concatenate([training_features, padding], axis=0)
#         training_features = new_features
#         new_features = None
        
#         label_padding = np.zeros((args.batch_size - training_residual_shape, 1))
#         new_labels = np.concatenate([training_labels, label_padding], axis=0)
#         training_labels = new_labels
#         new_labels = None
    
#     # For labels
#     if validation_residual_shape != 0:
#         padding = np.zeros((args.batch_size - validation_residual_shape, args.image_size, args.image_size, args.image_depth))
#         new_features = np.concatenate([validation_features, padding], axis=0)
#         validation_features = new_features
#         new_features = None
#         label_padding = np.zeros((args.batch_size - validation_residual_shape, 1))
#         new_labels = np.concatenate([validation_labels, label_padding], axis=0)
#         validation_labels = new_labels
#         new_labels = None
    
    
#     return (training_features.astype('uint8'), training_labels.astype('uint8')), \
#         (validation_features.astype('uint8'), validation_labels.astype('uint8')), labels_value

# # create training data
# def get_data(training_features, training_labels, validation_features, validation_labels):
#     # get training data
#     train_imgs = tf.constant(training_features)
#     train_labels = tf.constant(training_labels)

#     # get validation data
#     validation_imgs = tf.constant(validation_features)
#     validation_labels = tf.constant(validation_labels)

#     training_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
#     validation_data = tf.data.Dataset.from_tensor_slices((validation_imgs, validation_labels))

#     return training_data, validation_data

# (train_features, train_labels), (validataion_features, validataion_labels) = get_files('quickdraw_data')
# image_label_ds = get_data(train_features, train_labels, validataion_features, validataion_labels)


# print('image shape: ', image_label_ds.output_shapes[0])
# print('label shape: ', image_label_ds.output_shapes[1])
# print('types: ', image_label_ds.output_types)
# print()
# print(image_label_ds)


# Data Generator class
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# function to get the IDs and labels
def get_data(path):
    partition = {'train':[], 'validation':[]}
    labels = {}
    count = 0
    dirs = [x[0] for x in os.walk(path)][1:]
    for d in dirs:
        count += 1
        files = [f for f in os.listdir(d)]
        for f in files:
            if '.npy' in f:
                name = f.strip('.npy')
                if random.randint(1, 100) % 10 != 0:
                    partition['train'].append(name)
                else:
                    partition['validation'].append(name)
                labels[name] = count

    return partition, labels, count


# function to make images to npy files
def parse_imgs(path):
    imgs = ['.png', '.jpg', '.jpeg']
    dirs = [x[0] for x in os.walk(path)][1:]
    for d in dirs:
        files = [f for f in os.listdir(d)]
        for f in files:
            if any(st in f for st in imgs):
                img_raw = tf.io.read_file(d+'/'+f)
                img_tensor = tf.image.decode_image(img_raw)
                img_tensor = tf.image.resize(img_tensor, [1, args.image_size, args.image_size, args.image_depth])
                img_data = np.copy(img_tensor)
                for t in imgs:
                    f = f.strip(t)
                np.save(path + '/' + d + '/' + f + '.npy', img_data)