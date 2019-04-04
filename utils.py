import os
import tensorflow as tf
import numpy as np

# get all the files and its label
def get_files(path, args):
    dirs = [x[0] for x in os.walk(path)][1:]
    
    training_features = None
    validation_features = None
    training_labels = None
    validation_labels = None
    labels_value = []
    count = 1

    #TODO: Load Img
    for d in dirs:
        files = [f for f in os.listdir(d)] #load np
        for f in files:
            data = np.load(d+'/'+f)                
            data = data.reshape([-1, args.image_size, args.image_size, args.image_depth])
            length = data.shape[0]
            # concatenate arrays to get the training data
            if training_features is None:
                training_features = np.copy(data[:length-(length//10),:,:,:])
            else:
                training_features = np.concatenate([training_features, data[:length-(length//10),:,:,:]], axis=0)
            
            # get validation data
            if training_labels is None:
                training_labels = count * np.ones((length-(length//10),1))
            else:
                new_labels = count * np.ones((length-(length//10),1))
                training_labels = np.concatenate([training_labels, new_labels], axis=0)

            # concatenate arrays to get the validation data
            if validation_features is None:
                validation_features = np.copy(data[length-(length//10):,:,:,:])
            else:
                validation_features = np.concatenate((validation_features, data[length-(length//10):,:,:,:]), axis=0)

            # get validation data
            if validation_labels is None:
                validation_labels = count * np.ones(((length//10),1))
            else:
                new_labels = count * np.ones(((length//10),1))
                validation_labels = np.concatenate([validation_labels, new_labels], axis=0)
        
        labels_value.append(d)
        count += 1
    
    # Pad the data to fit the batch size
    # For features
    training_residual_shape = training_features.shape[0] % args.batch_size
    validation_residual_shape = validation_features.shape[0] % args.batch_size
    if training_residual_shape != 0:
        padding = np.zeros((training_residual_shape, args.image_size, args.image_size, args.image_depth))
        training_features = np.concatenate([training_features, padding], axis=0)
        label_padding = np.zeros((training_residual_shape, 1))
        training_labels = np.concatenate([training_labels, label_padding], axis=0)
    
    # For labels
    if validation_residual_shape != 0:
        padding = np.zeros((validation_residual_shape, args.image_size, args.image_size, args.image_depth))
        validation_features = np.concatenate([validation_features, padding], axis=0)
        label_padding = np.zeros((validation_residual_shape, 1))
        validation_labels = np.concatenate([validation_labels, label_padding], axis=0)
    
    
    return (training_features.astype('uint8'), training_labels.astype('uint8')), \
        (validation_features.astype('uint8'), validation_labels.astype('uint8'))

# create training data
def get_data(training_features, training_labels, validation_features, validation_labels):
    # get training data
    print(training_features.shape, training_labels.shape)
    train_imgs = tf.constant(training_features)
    train_labels = tf.constant(training_labels)

    # get validation data
    validation_imgs = tf.constant(validation_features)
    validation_labels = tf.constant(validation_labels)

    training_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
    validation_data = tf.data.Dataset.from_tensor_slices((validation_imgs, validation_labels))

    return training_data, validation_data

# (train_features, train_labels), (validataion_features, validataion_labels) = get_files('quickdraw_data')
# image_label_ds = get_data(train_features, train_labels, validataion_features, validataion_labels)


# print('image shape: ', image_label_ds.output_shapes[0])
# print('label shape: ', image_label_ds.output_shapes[1])
# print('types: ', image_label_ds.output_types)
# print()
# print(image_label_ds)
