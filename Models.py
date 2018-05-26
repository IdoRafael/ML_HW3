from ReadWrite import read_data

def load_prepared_data():
    return read_data('train.csv', index=True), read_data('validate.csv', index=True), read_data('test.csv', index=True)


def optimize_models_parameters():
    pass


def load_optimize_train_select_and_predict():
    train, validate, test = load_prepared_data()
    models = optimize_models_parameters()