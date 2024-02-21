from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from contracts import contract, new_contract

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
 