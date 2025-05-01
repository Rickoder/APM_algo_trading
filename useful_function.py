import numpy as np 


def get_pne(signal, returns):
    return (signal * returns).sum(axis=1)


def get_sr(pnl):
    return np.mean(pnl) / np.std(pnl) * np.sqrt(252)


def sequence_splitter(X_mat, target_col, prediction_steps,  predict_steps=1):
    """
    Keras wants 3d arrays (sample ##it is the number of series##, timesteps, number_of_features)
    We want to use this function in a loop. Where we will append series after series.
    
    X_mat:The dataframe where we do have our features 
    prediction_steps: Number of steps used to predict. It is also the number of hidden layers!
    Target_col: what do we want to predict 
    predict_steps: how far we want to predict
    """

    # We need the following to create the np array to feed.
    nb_seq = len(X_mat) - prediction_steps - predict_steps
    nb_feat = X_mat.shape[1]

    X = np.empty((nb_seq, prediction_steps, nb_feat))
    Y = np.empty((nb_seq, predict_steps))

    target = X_mat[target_col]

    for step in range(0,nb_seq): 
        X[step] = X_mat.iloc[step:prediction_steps+step,]
        Y[step] = target[step+1]

    return X,Y