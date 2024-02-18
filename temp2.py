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
    if not model.layers[0].input_shape is None:
        raise ContractException("Input shape not specified for the first layer.")
    
#excel sheet row 10
@new_contract
def batch_norm_order(model):
     dense_flag = False
     batch_flag = False

     for layer in model.layers:
        if isinstance(layer, layers.Dense):
            dense_flag = True
            batch_flag = False
        elif isinstance(layer, layers.BatchNormalization):
            if not dense_flag:
                raise ValueError("BatchNormalization must follow Dense layer.")
            batch_flag = True
        elif isinstance(layer, layers.Activation):
            if not dense_flag or batch_flag:
                raise ValueError("Activation must follow BatchNormalization which follows Dense layer.")
            

#excel sheet row 12
@new_contract
def contract_checker_PReLU(model):
    found_prelu = False

    for layer in model.layers[::-1]:
        if isinstance(layer, layers.PReLU):
            found_prelu = True
            break

    if found_prelu:
        print("PReLU layer added directly to the model using add() method.")
    else:
        print("PReLU layer may not be added directly using add() method or it's wrapped with Activation class.")

#excel sheet row 13
@new_contract
def contract_check_add_after_compile(model):
    if model.compiled:
        raise ContractException("Layers cannot be added to the model after compilation.")
