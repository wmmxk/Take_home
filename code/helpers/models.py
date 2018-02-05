
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

def create_GP(num_input):

    '''
    input: num_input: an integer describing the number of features 

    output: a Gaussian Process model
    '''

    kernel_Matern = 1.0 * Matern([1.0]*num_input)
    model = GaussianProcessRegressor(kernel = kernel_Matern, n_restarts_optimizer=1)

    return model



def create_FFN(num_input):
    '''
    input: num_input: an integer describing the number of features 

    output: a compiled neural network
    '''

    model = Sequential()
    
    model.add(Dense(num_input, input_dim = num_input))
    model.add(PReLU())
    model.add(BatchNormalization())
        
    model.add(Dense(220)) 
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.3))
    
    model.add(Dense(60))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(1))
    model.compile(loss = 'mae', optimizer = 'adam')
    return model

