model_train = EfficientCapsNet(model_name, mode='train', verbose=True)

dataset = Dataset(model_name, config_path='config.json')
history = model_train.train(dataset, initial_epoch=0)



class EfficientCapsNet(object):
    """
    A class used to manage an Efficiet-CapsNet model. 'model_name' and 'mode' define the particular architecure and modality of the 
    generated network.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    custom_path: str
        custom weights path
    verbose: bool
    
    Methods
    -------
    load_graph():
        load the network graph given the model_name
    train(dataset, initial_epoch)
        train the constructed network with a given dataset. All train hyperparameters are defined in the configuration file

    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    verbose: bool
    
    Methods
    -------
    load_config():
        load configuration file
    load_graph_weights():
        load network weights
    predict(dataset_test):
        use the model to predict dataset_test
    evaluate(X_test, y_test):
        comute accuracy and test error with the given dataset (X_test, y_test)
    save_graph_weights():
        save model weights

    """

    def __init__(self, model_name, mode='test', config_path='config.json', verbose=True):
        self.model_name = model_name
        self.model = None
        self.mode = mode
        self.config_path = config_path
        self.config = None
        self.verbose = verbose
        self.load_config()


    def load_config(self):
        """
        Load config file
        """
        with open(self.config_path) as json_data_file:
            self.config = json.load(json_data_file)
    

    def load_graph_weights(self):
        try:
            self.model.load_weights(self.model_path)
        except Exception as e:
            print("[ERRROR] Graph Weights not found")
            
        
    def predict(self, dataset_test):
        return self.model.predict(dataset_test)
    

    def evaluate(self, X_test, y_test):
        print('-'*30 + f'{self.model_name} Evaluation' + '-'*30)

        y_pred, X_gen =  self.model.predict(X_test)
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
        test_error = 1 - acc
        print('Test acc:', acc)
        print(f"Test error [%]: {(test_error):.4%}")
        print(f"N° misclassified images: {int(test_error*len(y_test))} out of {len(y_test)}")


    def save_graph_weights(self):
        self.model.save_weights(self.model_path)


    def __init__(self, model_name, mode='test', config_path='config.json', custom_path=None, verbose=True):
        Model.__init__(self, model_name, mode, config_path, verbose)
        if custom_path != None:
            self.model_path = custom_path
        else:
            self.model_path = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet_{self.model_name}.h5")
        self.model_path_new_train = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet{self.model_name}_new_train.h5")
        self.tb_path = os.path.join(self.config['tb_log_save_dir'], f"efficient_capsnet_{self.model_name}")
        self.load_graph()
    

    def load_graph(self):
        if self.model_name == 'MNIST':
            self.model = efficient_capsnet_graph_mnist.build_graph(self.config['MNIST_INPUT_SHAPE'], self.mode, self.verbose)

            
    def train(self, dataset=None, initial_epoch=0):
        callbacks = get_callbacks(self.tb_path, self.model_path_new_train, self.config['lr_dec'], self.config['lr'])

        if dataset == None:
            dataset = Dataset(self.model_name, self.config_path)
        dataset_train, dataset_val = dataset.get_tf_data()    


        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
            loss=[marginLoss, 'mse'],
            loss_weights=[1., self.config['lmd_gen']],
            metrics={'Efficient_CapsNet': 'accuracy'})
        steps=None

        print('-'*30 + f'{self.model_name} train' + '-'*30)

        history = self.model.fit(dataset_train,
          epochs=self.config[f'epochs'], steps_per_epoch=steps,
          validation_data=(dataset_val), batch_size=self.config['batch_size'], initial_epoch=initial_epoch,
          callbacks=callbacks)
        
        return history

