class Validation_Functions:
    #-----------------------------------------------------------------------------
    # Define custom loss functions for regression in Keras 
    #-----------------------------------------------------------------------------

    def __init__(self):
        pass

    # root mean squared error (rmse) for regression
    def rmse(self,y_true, y_pred):
        from keras import backend
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    # mean squared error (mse) for regression
    def mse(self,y_true, y_pred):
        from keras import backend
        return backend.mean(backend.square(y_pred - y_true), axis=-1)

    # coefficient of determination (R^2) for regression
    def r_square(self,y_true, y_pred):
        from keras import backend as K
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return (1 - SS_res/(SS_tot + K.epsilon()))

    def r_square_loss(self,y_true, y_pred):
        from keras import backend as K
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))
