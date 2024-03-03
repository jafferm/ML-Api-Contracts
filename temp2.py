from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from contracts import contract, new_contract
from keras import backend as K
import numpy as np
from contract_checker_library import ContractException  # Assuming you have a ContractException class in your library
## For Post 3
# Define a contract checking function for build_fn
@new_contract
def keras_model_contract(model):
    if not isinstance(model, Sequential):
        raise ContractException("The build_fn should return a Keras Sequential model.")

# Your nn_model function
def nn_model():
    return
    # ... your model creation logic here ...

# Contract checking function
@contract(build_fn='callable,keras_model_contract')
def create_nn_model():
    return nn_model()

# Wrap the neural network model function as a regressor using KerasRegressor
regressor = KerasRegressor(build_fn=create_nn_model, nb_epoch=2)
##For Post 12
# Define a contract checking function for model usage
@new_contract
def keras_model_usage_contract(model):
    if not isinstance(model, Sequential):
        raise ContractException("The model should be an instance of Keras Sequential.")
    if not model.compiled:
        raise ContractException("The model has not been compiled. Please compile the model before training.")
# Load weights and make predictions contract checking function
@contract(model='callable,keras_model_usage_contract', weights='str', input_data='array_like')
def load_weights_and_predict(model, weights, input_data):
    model.load_weights(weights)
    return model.predict(input_data)

# Excel sheet row 6
@new_contract
def reference_to_model_input_contract(ref):
    if ref != model.input:
        raise ContractException("Reference should be made to the model's input.")
    
# excel sheet row 7
@new_contract
def contract_checker1(model):
    concatenate_layers = [layer for layer in model.layers if isinstance(layer, Concatenate)]

    for concatenate_layer in concatenate_layers:
        if concatenate_layer.input_shape is None or None in concatenate_layer.input_shape:
            raise ContractException("Input shape not specified for a Concatenate layer.")

#excel sheet row 10
@new_contract
def batch_norm_order(model):
    for i in range(1, len(model.layers) - 1):  # Iterate from the second layer to the second-to-last layer
     current_layer = model.layers[i]
     previous_layer = model.layers[i - 1]
     next_layer = model.layers[i + 1]

     if isinstance(current_layer, BatchNormalization):
            if isinstance(previous_layer, Dense) and not isinstance(next_layer, Dense):
                break
            else:
                    raise ContractException("Invalid layer configuration: The layer before Batch Normalization should be Dense(linear-layer), and the layer after should be non-linear.")
            

#excel sheet row 12
@new_contract
def contract_checker_PReLU(model):
    for layer in model.layers:
        if isinstance(layer, PReLU):
        # Check if PReLU layer is wrapped with an activation layer
            if len(layer._layers) > 0 and isinstance(layer._layers[0], Activation):
                raise ContractException("PReLU layer is wrapped with an Activation layer.")

#excel sheet row 16
@new_contract           
def contract_check_sequential_model(model,input_data, target_data):
    # Check if the model's output dimensions match target data dimensions
    msg1=""
    msg2=""
    model_output = model.predict(input_data)
    if model_output.shape[1] != target_data.shape[1]:
        # Check if the LSTM layer has return sequence set to true
        lstm_layers = [layer for layer in model.layers if isinstance(layer, LSTM)]

        for lstm_layer in lstm_layers:
            if not lstm_layer.return_sequences:
                msg1+="LSTM layer {lstm_layer.name} does not have return_sequences set to True."
                
        # Check if the Dense layer is wrapped in TimeDistributed
        dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]

        for dense_layer in dense_layers:
             if not any(isinstance(wrapper, TimeDistributed) for wrapper in dense_layer._layers):
                msg2+="Dense layer {dense_layer.name} is not wrapped in TimeDistributed."
    raise ContractException(msg1+msg2)
 
#this week
 #excel sheet row 17
@new_contract
def contract_check_concat_parameters(input_b, intermediate_from_a):
    # Get the output shapes of the two layers
    if isinstance(intermediate_from_a, np.ndarray):
        raise ContractException("intermediate_from_a is a NumPy array. Do not use predict()")
    else:
        shape1 = K.int_shape(input_b)
        shape2 = K.int_shape(intermediate_from_a)

        # Check if the dimensions are compatible for concatenation
        if shape1 and shape2 and shape1[1] == shape2[1]:
            return True
        else:
            raise ContractException("The dimensions of the two layers are not compatible for concatenation.")

#excel sheet row 19 
@new_contract
def contract_cnn_with_lstm(model):
    # Check if the model has a convolutional layer followed by an LSTM layer
    for i in range(0, len(model.layers) - 1):  # Iterate from the first layer to the second-to-last layer
     current_layer = model.layers[i]
     next_layer = model.layers[i + 1]
     if isinstance(current_layer, Conv2D):
            if isinstance(next_layer, LSTM):
                cnn_output_shape = current_layer.output_shape
                lstm_input_shape = next_layer.input_shape
                # Check if the output shape of the CNN and input shape of the LSTM are compatible
                if cnn_output_shape[-1] != lstm_input_shape[-1]:
                    raise ContractException("Output shape of Conv2D do not match input shape of LSTM, Use 'TimeDistributed' to wrapper on the CNN layer.")

#excel sheet row 22
@new_contract
def check_reset_weights(model):
    initial_weights = tf.keras.models.load_model('initial_weights.h5').get_weights()
    #This loop iterates over the layers of the model and the corresponding initial weights loaded from the saved file.
    for layer, initial_weight in zip(model.layers, initial_weights):
            current_weight = layer.get_weights()
            #This line checks whether all elements of the current_weight tensor are equal to the initial_weight tensor using TensorFlow operations.
            if not tf.reduce_all(tf.equal(current_weight, initial_weight)):
                raise ContractException("Weights are not reset to initial weights.")

#excel sheet row 25
@new_contract
def check_BN_updateOps(model, X_train, y_train):
    # Start a TensorFlow session
    with tf.compat.v1.Session() as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # Get the update operations from model
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Run the update operations and the forward pass
        sess.run([update_ops, model.output], feed_dict={model.input: X_train, model.output: y_train})

        # Check if update operations were executed
        if not update_ops:
            raise ContractException("Batch Normalization statistics are not updated during training. "
                                    "You need to manually add the update operations.")    
    
#excel sheet row 65
@new_contract
def check_mergeLayer_input(model):
    for layer in model.layers:
        if isinstance(layer, Merge):
            # Access the inputs property of the merge layer
            input_types = layer.inputs

            # Check if the input types are instances of Keras models
            for input_type in input_types:
                if  isinstance(input_type, Model):
                    raise ContractException("Use functional API merge layers Add() or substract()"
                                             " to merge output of two models")
                
#excel sheet row 72
@new_contract
def check_multi_initialization(model):
    if K.backend() == 'tensorflow':
        raise ContractException("The Backend used is tensorflow, please use clear_session()"
                                " after usage of model in loop ")
    
#excel sheet row 81
@new_contract
def check_conv1d_input_shape(model):
    for layer in model.layers:
        if isinstance(layer, Conv1D):
            if layer.input_shape is not None and (len(layer.input_shape) != 3):
                raise ContractException("The layer does not have a spatial dimension in its input shape. "
                                "Expected input shape (batch_size, steps, features).")
            
#excel sheet row 84
@new_contract
def check_conv2d_input_shape(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            if layer.input_shape is not None and (len(layer.input_shape) != 3):
                raise ContractException("input_shape should contain 3 dimensions only,"
                                        " Expected input shape (height, width, channels).")

#excel sheet row 89
@new_contract
def check_embedding_argument(model,data):
    # Create a Tokenizer
    tokenizer = Tokenizer()
    # Fit the tokenizer on the text data
    tokenizer.fit_on_texts(data)
    # Vocabulary size
    vocab_size = len(tokenizer.word_counts) + 1  # Add 1 for the reserved index
    for layer in model.layers:
        if isinstance(layer, Embedding):
            # Retrieve the configuration of the Embedding layer
            embedding_config = layer.get_config()
            # Access the first argument (input_dim) from the configuration
            input_dim = embedding_config['input_dim']
            if input_dim != vocab_size:
                raise ContractException("The first argument provided to the Embedding layer"
                                        " should be the vocabulary size+1.")
            
#excel sheet row 92 
@new_contract
def check_lstm_input_shape(model):
    
    for layer in model.layers:
        if isinstance(layer, LSTM):
            input_shape = layer.input_shape
            if len(input_shape) != 3:
                raise ContractException("Invalid input shape for LSTM layer. "
                                        "Expected input shape (batch_size, timesteps, input_dim), ")
            
#excel sheet row 99
@new_contract
def check_rnn_input_shape(model):
    for layer in model.layers:
        if isinstance(layer, keras.layers.recurrent.RNN):
            # Access the input shape of the RNN layer
            input_shape = layer.input_shape
            if len(input_shape) == 2:
                # Check if num_features are specified
                num_timesteps, num_features = input_shape
                if  num_features is None:
                    raise ContractException("The num_features should be specified in the input shape.")
            elif len(input_shape) == 3:
                # Check if num_features are specified
                num_samples,num_timesteps, num_features = input_shape
                if  num_features is None:
                    raise ContractException("The num_features should be specified in the input shape.")
            else:
                raise ContractException("Invalid input shape for RNN layer. "
                                        "Expected input shape (num_timesteps, num_features) or" 
                                        "(num_samples, num_timesteps, num_features), ")
            
#excel sheet row 103
@new_contract
def check_model_input_shape(model):
    if isinstance(model, Sequential):
        layer=model.layers[0]
        # Check if the input_shape attribute exists and is not None
        if not hasattr(layer, 'input_shape') or layer.input_shape is None:
            raise ContractException("The input shape of the first layer should be explicitly specified.")

#excel sheet row 104
@new_contract
def check_add_method(model):
    """checks whether the add method is present and callable,
       which would indicate that layers have been added to the model using the add method.
    """
    if hasattr(model, 'add') and callable(getattr(model, 'add', None)):
        if not isinstance(model, Sequential):
            raise ContractException("In order to add layers to a model, using add method the model should "
                                     "be a Sequential model. In case of loading a saved model, create a "
                                     "new Sequential model then add loaded model, then you can use add method"
                                      " to add layers") 
        
#excel sheet row 105
@new_contract
def check_load_weights_method(model):
    """
    Check if the model variable has the load_weights method call like model.load_weights().
    Then check if the variable is a Keras Model before trying to load weights.

    Parameters:
    - model: The loaded model object.
    """
    if hasattr(model, 'load_weights') or callable(getattr(model, 'load_weights', None)):
        if not isinstance(model, Model):
            raise ContractException("In order to load weights to a model, the model should be a Keras Model."
                                    "Try to create or load the model before loading weights.")

#excel sheet row 109
@new_contract
def is_custom_loss(model):
    # Get the compile arguments of the model
    compile_args = model.compile_args

    # Check if 'loss' key is present and corresponds to a callable object
    if 'loss' in compile_args and callable(compile_args['loss']):
        return True
    else:
        raise ContractException("If the loss function is a custom loss function, "
                                "then pass it as a function object, not a string.")

#excel sheet row 113: 
@new_contract
def check_confusion_matrix_input(y_test, y_pred):
    """
    Check if the input to confusion_matrix is in the correct format.

    checks if the inputs have more than one dimension, which is a common 
    characteristic of one-hot encoded arrays.
    """
    if y_test.ndim != 1 or y_pred.ndim != 1:
        raise ContractException("Input must be an array of int, not one hot encodings")
    #check ifinput is an integer array
    if not np.issubdtype(y_test.dtype, np.integer) and np.issubdtype(y_pred.dtype, np.integer):
        raise ContractException("Input must be an array of int .")
    
#excel sheet row 114:
@new_contract
def check_evaluate_assignment(result):
    """
    Check if the user has assigned a pair of variables to the result of 'model.evaluate'.
    """
    if isinstance(result, tuple) and len(result) > 1:
        raise ContractException("Assigning a pair of variables to the result of 'model.evaluate' is not allowed. "
                                "Use a single variable to capture the result.")

#excel sheet row 121
@new_contract
def check_input_shape(model, X):
    """
    Check if the input data shape matches the expected input shape of the model.

    Parameters:
        - model: The Keras model.
        - X: The input data.

    Raises:
        - ContractException: If the input data shape does not match the expected input shape.
    """
    expected_input_shape = model.input_shape[1:]  # Exclude batch size
    actual_input_shape = X.shape[1:]

    if expected_input_shape != actual_input_shape:
        raise ContractException(f"Input data shape {actual_input_shape} does not match "
                                f"the expected input shape {expected_input_shape} of the model.")

#excel sheet row 126
@new_contract
def check_normalized_input(data):
    """
    Check if the input data is properly normalized.

    """
    # Check if the data is a numpy array
    if not isinstance(data, np.ndarray):
        raise ContractException("Input data must be a NumPy array.")
    
    # Check if the data is in the range [0, 1]
    if not (np.min(data) >= 0 and np.max(data) <= 1):
        raise ContractException("Input data must be normalized between 0 and 1.")
    
#excel sheet row 143
@contract
def check_build_fn_parameter(classifier: KerasClassifier):
    """
    Check the build_fn parameter of a KerasClassifier.
    """
    build_fn = classifier.build_fn

    if not callable(build_fn):
        raise ContractException("build_fn must be a callable (function or class instance).")

    if inspect.isclass(build_fn):
        # If build_fn is a class, check if it has a __call__ method
        if not hasattr(build_fn, '__call__'):
            raise ContractException("If build_fn is a class, it must have a __call__ method.")

        # Check if the instance of the class returns a compiled Keras model
        instance = build_fn()
        if not isinstance(instance, Model):
            raise ContractException("The __call__ method of the build_fn class must return a compiled Keras model.")
    elif not isinstance(build_fn(), Model):
        raise ContractException("The build_fn function must return a compiled Keras model.")

    
    
