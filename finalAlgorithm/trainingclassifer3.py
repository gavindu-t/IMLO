import hyperopt
from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


def objective_fun(params):
    # defining the initial version of the neural network
    model = Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(params['hiddenLayerOne'],
              activation=params['activation'], input_dim=124))
    model.add(layers.Dropout(params['dropout']))
    model.add(layers.Dense(params['hiddenLayerTwo'],
              activation=params['activation']))
    model.add(layers.Dropout(params['dropout']))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer=params['optimizer'](params['learning_rate']),
                  loss='categorical_crossentropy', metrics='accuracy')

    input_shape = X_train.shape
    model.build(input_shape)
    es = EarlyStopping(monitor='val_loss', mode='min',
                       verbose=1, patience=15)

    model.fit(X_train, y_train, validation_data=(X_val, y_val))

    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc,
            'status': STATUS_OK,
            'model': model,
            'params': params}


param_space = {
    "activation": hp.choice("activation", ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                                           'elu', 'exponential', 'softmax']),
    "optimizer": hp.choice("optimizer", [SGD, Adam, RMSprop, Adagrad, Adamax, Nadam, Ftrl]),
    "learning_rate": hp.uniform("learning_rate", 0.001, 1),
    "epochs": hp.uniform("epochs", 10, 100),
    "hiddenLayerOne": hp.uniform("hiddenLayerOne", 10, 100),
    "hiddenLayerTwo": hp.uniform("hiddenLayerTwo", 10, 100),
    "dropout": hp.choice("dropout", [0.1, 0.4, 0.6])
}

model.add(layers.Dropout(params['dropout']))

trials = Trials()

best_params = fmin(
    fn=objective_fun,
    space=param_space,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials)
