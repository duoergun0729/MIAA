import numpy as np
np.random.seed(1000)
import imp 
import input_data_class
import keras
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import tensorflow as tf
import os
import configparser
import argparse
from keras import regularizers

import logging
import logging.config

logging.config.fileConfig('log.ini')
logger = logging.getLogger('api')

parser = argparse.ArgumentParser()
parser.add_argument('-dataset',default='location')
parser.add_argument('-e',default=0.1)
parser.add_argument('-l2', action='store_true', default=False)
parser.add_argument('-dropout', action='store_true', default=False)
args = parser.parse_args()
dataset=args.dataset 
input_data=input_data_class.InputData(dataset=dataset)
config = configparser.ConfigParser()
config.read('config.ini')

num_classes=int(config[dataset]["num_classes"])
save_model=True
user_epochs=int(config[dataset]["user_epochs"])
batch_size=int(config[dataset]["batch_size"])
result_folder=config[dataset]["result_folder"]
network_architecture=str(config[dataset]["network_architecture"])
fccnet=imp.load_source(str(config[dataset]["network_name"]),network_architecture)

print(fccnet)

print("dataset: {}".format(dataset))
print("epochs: {}".format(user_epochs))
print("result folder: {}".format(result_folder))
print("network architecture: {}".format(network_architecture))

####### you may need to comment the code if not use GPU
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5
config_gpu.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config_gpu))

(x_train,y_train),(x_test,y_test) =input_data.input_data_user()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)
y_train=y_train.astype(int)
y_test=y_test.astype(int)

#print(x_train)
#print(y_train)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape=x_train.shape[1:]

if args.l2:
    model=fccnet.model_user_l2(input_shape=input_shape,labels_dim=num_classes,e=float(args.e))
elif args.dropout:
    model=fccnet.model_user_dropout(input_shape=input_shape,labels_dim=num_classes,e=float(args.e))
    #model=fccnet.model_user(input_shape=input_shape,labels_dim=num_classes)
else:
    model=fccnet.model_user(input_shape=input_shape,labels_dim=num_classes)

lr=0.01

#set opt
opt=keras.optimizers.SGD(lr=lr)

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])    
model.summary()

index_array=np.arange(x_train.shape[0])
batch_num=np.int(np.ceil(x_train.shape[0]/batch_size))
for i in np.arange(user_epochs):
    np.random.shuffle(index_array)
    for j in np.arange(batch_num):
        x_batch=x_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,x_train.shape[0])],:]
        y_batch=y_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,x_train.shape[0])],:]
        model.train_on_batch(x_batch,y_batch)  
        
    if  (dataset == "location") :   
        if (i+1)%150==0:
            #decay the learning rate by 0.1 
            K.set_value(model.optimizer.lr,K.eval(model.optimizer.lr*0.1))
            print("Learning rate: {}".format(K.eval(model.optimizer.lr)))
    if  (dataset == "CH-MNIST")  :   
        if (i+1)%350==0:
            #decay the learning rate by 0.1 
            K.set_value(model.optimizer.lr,K.eval(model.optimizer.lr*0.1))
            print("Learning rate: {}".format(K.eval(model.optimizer.lr)))
    
    if (i+1)%100==0:
        print("Epochs: {}".format(i))
        scores_test = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', scores_test[0])
        print('Test accuracy:', scores_test[1])  
        scores_train = model.evaluate(x_train, y_train, verbose=0)
        print('Train loss:', scores_train[0])
        print('Train accuracy:', scores_train[1])  

        
logger.debug("[Target Model]Test loss:{}".format(scores_test[0]))   
logger.debug("[Target Model]Test accuracy:{}".format(scores_test[1]))  

        
##save the model
if save_model:
    weights=model.get_weights()
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(result_folder+"/models"):
        os.makedirs(result_folder+"/models")
        
    if args.l2:    
        np.savez(result_folder+"/models/"+"epoch_{}_weights_user_L2_{}.npz".format(user_epochs,args.e),x=weights)
    elif args.dropout:    
        np.savez(result_folder+"/models/"+"epoch_{}_weights_user_dropout_{}.npz".format(user_epochs,args.e),x=weights)
    else:
        np.savez(result_folder+"/models/"+"epoch_{}_weights_user.npz".format(user_epochs),x=weights)