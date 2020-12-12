# U-Net Model Construction

import unittest

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l1, l2


def segmentation_unet(input_size, num_classes, filter_multiplier=10, regularization_rate=0.):
    input_ = Input((input_size, input_size, 1))
    skips = []
    output = input_

    num_layers = 4#int(np.floor(np.log2(input_size))) #do not change (could add more if every other layer does not increase the image size)
    down_conv_kernel_sizes = np.zeros([num_layers], dtype=int)
    down_filter_numbers = np.zeros([num_layers], dtype=int) #could change
    up_conv_kernel_sizes = np.zeros([num_layers], dtype=int) #do not change
    up_filter_numbers = np.zeros([num_layers], dtype=int) #could change
    
    
    #this for loop is good for modifications.
    for layer_index in range(num_layers):
        down_conv_kernel_sizes[layer_index] = int(3)
        down_filter_numbers[layer_index] = (2**(layer_index)) * 16  #typical rule is the deeper you are the more filters. But there is no "real" method. Starting with 8 actually has meaning to check to see if yuo are going to each different pixels new line. You can double each layer (at deeper layers). num classes = 2
        up_conv_kernel_sizes[layer_index] = int(2) #why is it four? Read up on this.
        up_filter_numbers[layer_index] = 2**(num_layers - layer_index - 1) * 8 - 8 + num_classes
        #have to make sure that in the final layer hte number of filters is two.
    '''
    
    
    for layer_index in range(num_layers):
        down_conv_kernel_sizes[layer_index] = int(3)
        down_filter_numbers[layer_index] = int((layer_index + 1) * filter_multiplier + num_classes) #typical rule is the deeper you are the more filters. But there is no "real" method. Starting with 8 actually has meaning to check to see if yuo are going to each different pixels new line. You can double each layer (at deeper layers). num classes = 2
        up_conv_kernel_sizes[layer_index] = int(4) #why is it four? Read up on this.
        up_filter_numbers[layer_index] = int((num_layers - layer_index - 1) * filter_multiplier + num_classes)
        #have to make sure that in the final layer hte number of filters is two.
    '''
    #later on (after you mess with the above)
    count = 0
    for shape, filters in zip(down_conv_kernel_sizes, down_filter_numbers):
         
        output = Conv2D(filters, (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(1,1),  activation="relu"
                        )(output) #, bias_regularizer=l1(regularization_rate),  
        output = BatchNormalization()(output)
        output = SpatialDropout2D(0.3)(output)
        output = Conv2D(filters, (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(1,1),  activation="relu"
                        )(output) #, bias_regularizer=l1(regularization_rate),  
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size = (2,2))(output)
        skips.append(output)
        count += 1
        
    output = Conv2D(filters, (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(2,2),  activation="relu"
                        )(output)
    print([output])
    count = 0
    for shape, filters in zip(up_conv_kernel_sizes, up_filter_numbers):
        #output = UpSampling2D()(output)
        output = Conv2DTranspose(filters, (shape,shape), strides=(2,2), padding='same')(output)
        skip_output = skips.pop()
        output = concatenate([output, skip_output], axis=3)
        output = Conv2D(filters, (shape, shape), activation="relu", padding='same', strides = (1,1),
                            bias_regularizer=l1(regularization_rate))(output)
        output = BatchNormalization(momentum=.9)(output)
        output = Conv2D(filters, (shape, shape), activation="relu", padding='same', strides = (1,1),
                            bias_regularizer=l1(regularization_rate))(output)
        output = BatchNormalization(momentum=.9)(output)
        
    output = Conv2DTranspose(filters, (shape,shape), strides=(2,2), padding='same')(output)
    output = Conv2D(filters, (shape, shape), strides = (1,1), activation="softmax", padding="same", 
                            bias_regularizer=l1(regularization_rate))(output)
    '''
    for shape, filters in zip(up_conv_kernel_sizes, up_filter_numbers):
        print("UPSAMPLE")
        #output = UpSampling2D()(output)
        output = Conv2DTranspose(filters, (shape,shape), strides=(2,2), padding='same')(output)
        skip_output = skips.pop()
        output = concatenate([output, skip_output], axis=3) #axis = 3
        if filters != num_classes:
            output = Conv2D(filters, (shape, shape), activation="relu", strides = (2,2),-
                            bias_regularizer=l1(regularization_rate))(output)
            output = BatchNormalization(momentum=.9)(output)
        else:
            print("occurance of softmax")
            output = Conv2D(filters, (shape, shape), activation="softmax", padding="same", 
                            bias_regularizer=l1(regularization_rate))(output)
    '''
    assert len(skips) == 0
    return Model([input_], [output])


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    """
    weights = tf.keras.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)

        return loss

    return loss


class SagittalSpineUnetTest(unittest.TestCase):
    def test_create_model(self):
        model = segmentation_unet(128, 2)

if __name__ == '__main__':
    unittest.main()
