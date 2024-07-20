1. Install TensorFlow and Keras:
    
    ```bash
    
    pip install tensorflow
    pip install keras
    
    ```
    
    - These commands install TensorFlow and Keras, which are necessary for building and training the CNN.
2. Import TensorFlow and Keras modules:
    
    ```python
    
    import tensorflow as tf
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
    
    ```
    
    - `tensorflow` is imported for building and training the neural network.
    - `ImageDataGenerator` is imported from Keras for image data preprocessing and augmentation.

### Data Processing

1. Create training data generator with augmentation:
    
    ```python
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    ```
    
    - `rescale=1./255`: Scales the pixel values to the range [0, 1].
    - `shear_range=0.2`: Applies random shear transformations.
    - `zoom_range=0.2`: Applies random zoom-in augmentations.
    - `horizontal_flip=True`: Randomly flips images horizontally.
2. Load and preprocess training data:
    
    ```python
    
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')
    
    ```
    
    - Loads images from the directory `'dataset/training_set'`.
    - `target_size=(64, 64)`: Resizes all images to 64x64 pixels.
    - `batch_size=32`: Processes 32 images at a time.
    - `class_mode='binary'`: Uses binary labels for the classification.
3. Create test data generator without augmentation:
    
    ```python
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    ```
    
    - Only rescales the pixel values to [0, 1].
4. Load and preprocess test data:
    
    ```python
    
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')
    
    ```
    
    - Similar to training set but for test data in `'dataset/test_set'`.

### Building the CNN

1. Initialize the CNN:
    
    ```python
    
    cnn = tf.keras.models.Sequential()
    
    ```
    
    - Initializes a sequential model which allows building the CNN layer by layer.
2. Add the first convolutional layer:
    
    ```python
    
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    
    ```
    
    - `filters=32`: Number of filters (output channels).
    - `kernel_size=3`: Size of the convolutional kernel (3x3).
    - `activation='relu'`: Activation function ReLU.
    - `input_shape=[64, 64, 3]`: Shape of input images (64x64 pixels with 3 color channels).
3. Add the first pooling layer:
    
    ```python
    
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    ```
    
    - `pool_size=2`: Size of the pooling window (2x2).
    - `strides=2`: Steps to slide the pooling window.
4. Add the second convolutional layer:
    
    ```python
    
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    ```
    
    - Similar to the first convolutional and pooling layers but without specifying the input shape.
5. Flatten the feature maps:
    
    ```python
    pythonCopy code
    cnn.add(tf.keras.layers.Flatten())
    
    ```
    
    - Converts the 2D feature maps into a 1D feature vector.
6. Add the fully connected layer:
    
    ```python
    
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    
    ```
    
    - `units=128`: Number of neurons in the dense layer.
    - `activation='relu'`: Activation function ReLU.
7. Add the output layer:
    
    ```python
    
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    ```
    
    - `units=1`: Single output neuron for binary classification.
    - `activation='sigmoid'`: Activation function sigmoid.
8. Compile the CNN:
    
    ```python
    
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ```
    
    - `optimizer='adam'`: Optimizer used for training.
    - `loss='binary_crossentropy'`: Loss function for binary classification.
    - `metrics=['accuracy']`: Metric to evaluate the model.

### Training the CNN

1. Install necessary packages:
    
    ```bash
    
    pip install pillow
    pip install scipy
    
    ```
    
    - `pillow` is required for image processing.
    - `scipy` provides scientific computing tools.
2. Train the CNN:
    
    ```python
    
    from PIL import Image
    from tensorflow.keras.models import load_model
    cnn.fit(x=training_set, validation_data=test_set, epochs=25)
    
    ```
    
    - Trains the model on the training data and evaluates it on the test data over 25 epochs.

### Making a Single Prediction

1. Load and preprocess a single image:
    
    ```python
    
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('C:/Users/Himavanth Reddy/Desktop/DS/Machine Learning-A-Z-Codes-Datasets/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Python/dataset/dataset/single/dog.1354.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    ```
    
    - Loads an image from the specified path and resizes it to 64x64 pixels.
    - Converts the image to an array.
    - Expands the dimensions of the array to match the input shape expected by the CNN.
2. Make a prediction:
    
    ```python
    
    result = cnn.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print(prediction)
    
    ```
    
    - Uses the trained model to predict the class of the image.
