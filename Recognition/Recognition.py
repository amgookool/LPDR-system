from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import cv2
import os
import numpy as np
import pandas as pd

from keras.applications import resnet_v2
from keras.applications import efficientnet_v2
from keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# dataset_path = os.path.join("Recognition","English","Fnt")
# model_path = os.path.join("Recognition","Model")

# Set GPU device
# cv2.cuda.setDevice(0)

class RECOG_ALGO:
    
    class ALEXNET:
        def __init__(self,num_classes, input_shape) -> None:
            self.num_classes = num_classes
            self.input_shape = input_shape
            self.model_path = os.path.join("Recognition","Models","AlexNet","Model")
        
        def algorithm(self) -> Sequential:       
            model  = Sequential()
            # 1st conv layer
            model.add(Conv2D(filters=96, input_shape = self.input_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
            
            # 2nd conv layer
            model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
            model.add(Activation('relu'))
            # Max Pooling
            model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
            
            # 3rd conv layer
            model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
            model.add(Activation('relu'))
            
            # 4th conv layer
            model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
            model.add(Activation('relu'))
            
            # 5th conv layer
            model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
            model.add(Activation('relu'))
            # Max Pooling
            model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
            
            # Fully Connected Layer
            model.add(Flatten())
            
            # 1st Fully Connected Layer
            model.add(Dense(4096))
            model.add(Activation('relu'))
            
            # Add Dropout to prevent overfitting
            model.add(Dropout(0.5))
            
            # 2nd fully connected layer
            model.add(Dense(4096))
            model.add(Activation('relu'))
            # Add Dropout to prevent overfitting
            model.add(Dropout(0.5))
            
            # Output Layer
            model.add(Dense(self.num_classes))
            model.add(Activation('softmax'))
            
            model.compile(
                loss = 'categorical_crossentropy', 
                optimizer = 'adam', 
                metrics = ['accuracy']
            )
            model.summary()
            return model
            
    class RESNET:
        def __init__(self,num_classes, input_shape) -> None:
            self.num_classes = num_classes
            self.input_shape = input_shape
            self.model_path = os.path.join("Recognition","Models","ResNet","Model")
        
        def algorithm(self) -> Model:
            model = resnet_v2.ResNet50V2(include_top=True, weights=None,input_shape=self.input_shape,classes=self.num_classes)
            model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
            model.summary()
            return model
    
    class EfficientNet:
        def __init__(self,num_classes, input_shape) -> None:
            self.num_classes = num_classes
            self.input_shape = input_shape
            self.model_path = os.path.join("Recognition","Models","EfficientNet","Model")
        
        def algorithm(self):
            model = efficientnet_v2.EfficientNetV2B0(include_top=True, weights=None,input_shape=self.input_shape,classes=self.num_classes)
            model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
            model.summary()
            return model
       
    dataset_path = os.path.join("Recognition","Chars-Dataset","Fnt")
    alexnet_model_path = os.path.join("Recognition","Models","AlexNet")
    resnet_model_path = os.path.join("Recognition","Models","ResNet")
    efficientNet_model_path = os.path.join("Recognition","Models","EfficientNet")
    
    def train(self,model_name:str,epochs:int, height:int=128, width:int=128,depth:int=3, batch_size:int=128):
        directory = self.dataset_path
        
        data_generator = ImageDataGenerator(validation_split=0.2, rescale=1./255)
        
        train_generator = data_generator.flow_from_directory(
            directory=directory,
            subset="training",
            target_size=(width,height),
            batch_size=batch_size
        )
        
        validation_generator = data_generator.flow_from_directory(
            directory=directory,
            subset="validation",
            target_size= (width,height),
            batch_size=batch_size
        )
        
        num_classes = len(train_generator.class_indices)
        input_shape = (height,width,depth)
        
        if "alexnet" in model_name:
            checkpoint_path = os.path.join(self.alexnet_model_path,"Checkpoints")
            logs_paths = os.path.join(self.alexnet_model_path,"Logs")
            checkpoint_file = os.path.join(checkpoint_path,"model--epoch{epoch:02d}.h5")
            # checkpoint_file = os.path.join(checkpoint_path,"model--{epoch:02d}--{val_acc:.2f}.h5")
            csv_logger_file = os.path.join(self.alexnet_model_path,"training_logs.csv")
            model = self.ALEXNET(num_classes=num_classes,input_shape=input_shape).algorithm()
        
        if "resnet" in model_name:
            checkpoint_path = os.path.join(self.resnet_model_path,"Checkpoints")
            logs_paths = os.path.join(self.resnet_model_path,"Logs")
            checkpoint_file = os.path.join(checkpoint_path,"model--epoch{epoch:02d}.h5")
            # checkpoint_file = os.path.join(checkpoint_path,"model--{epoch:02d}--{val_acc:.2f}.h5")
            csv_logger_file = os.path.join(self.resnet_model_path,"training_logs.csv")
            model = self.RESNET(num_classes=num_classes, input_shape=input_shape).algorithm()
        
        if "efficientnet" in model_name:
            checkpoint_path = os.path.join(self.efficientNet_model_path,"Checkpoints")
            logs_paths = os.path.join(self.efficientNet_model_path,"Logs")
            checkpoint_file = os.path.join(checkpoint_path,"model--epoch{epoch:02d}.h5")
            # checkpoint_file = os.path.join(checkpoint_path,"model--{epoch:02d}--{val_acc:.2f}.h5")
            csv_logger_file = os.path.join(self.efficientNet_model_path,"training_logs.csv")
            model = self.EfficientNet(num_classes=num_classes, input_shape=input_shape).algorithm()
            
        with tf.device('/device:GPU:0'):
            # Setup Callbacks for Checkpoint epochs
            save_callback = ModelCheckpoint(
                filepath=checkpoint_file,
                monitor= 'val_accuracy',#'val_loss', 
                verbose=1,
                save_best_only=True,
                mode='auto'
            )

            # Setup callback to reduce learning rate as epochs progress
            reduce_lr = ReduceLROnPlateau(
                monitor='loss', 
                factor=0.2,
                patience=10, 
                min_lr=0.001
            )
            
            # Setup callback to stop training if loss doesn't decrease
            stop_callback = EarlyStopping(
                monitor='val_loss',
                verbose=1,
                patience=50
            )
            
            # Setup for TensorBoard Visualizations
            tensorboard = TensorBoard(log_dir=logs_paths)
            
            # Setup CSV Logger to log every epoch to a csv file
            csv_callback = CSVLogger(
                filename=csv_logger_file,
                separator=",",
                append=False
            )
            
            callback_list = [reduce_lr, save_callback, tensorboard, csv_callback, stop_callback]
            
            model.fit(
                train_generator, 
                steps_per_epoch= train_generator.n//train_generator.batch_size, 
                epochs=epochs,
                callbacks=callback_list,
                validation_data = validation_generator, 
                validation_steps = validation_generator.n//validation_generator.batch_size)
        
    def inference(self,image,model_name:str,generate_csv:bool=False):
        char = None
        chars = [
        '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H',
        'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
        ]
        
        if "alexnet" in model_name:
            model_path = os.path.join(self.ALEXNET(None,None).model_path,model_name)
        elif "resnet" in model_name:
            model_path = os.path.join(self.RESNET(None,None).model_path,model_name)
        else:
            model_path = os.path.join(self.EfficientNet(None,None).model_path,model_name)
        
        width,height,depth = 128,128,3
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        
        gpu_img = cv2.cuda.resize(gpu_img,(width,height))
        src = gpu_img.download()
        
        src = src.astype('float')/255.0
        src = img_to_array(src)
        src = np.expand_dims(src, axis=0)
        
        with tf.device('/device:GPU:0'):
            model = load_model(model_path, compile=False)
            # Get all prediction confidences for every class
            predictions = model.predict(src)
            
            def generate_predictions_df():
                """This function generates a csv file of the predictions made by the model.
                        Two files are created:
                            - test.csv: Contains all the predictions of all the classes
                            - test_top3.csv: Contains  the top 3 predictions of the classes.
                """
                predictions_df = pd.DataFrame(predictions)
                predictions_df = predictions_df.T 
                predictions_df = predictions_df.rename(columns={0:"Confidence"})    
                predictions_df = predictions_df.assign(Character=chars)
                
                predictions_df["Confidence"] = predictions_df["Confidence"].multiply(100)
                
                top_predictions = predictions_df.nlargest(n=5, columns="Confidence")
                top_predictions.to_csv(os.path.join("Recognition","test.csv"))
            
            if generate_csv:
                generate_predictions_df()
            
            predictions_df = pd.DataFrame(predictions).T
            predictions_df = predictions_df.rename(columns={0:"Confidence"}).assign(Character=chars)    
            predictions_df = predictions_df.sort_values(by=['Confidence'], ascending=False)
            prediction_dict : dict = predictions_df.to_dict()
            
            for char_idx in prediction_dict.get('Confidence'):
                confidence_score = prediction_dict['Confidence'].get(char_idx)
                if confidence_score >= 0.95:
                    char = prediction_dict['Character'].get(char_idx) 
                    return char
                else:
                    continue

# if __name__ == "__main__":
#     from time import time
#     start = time()
#     recognition_ai = RECOG_ALGO()
#     # recognition_ai.train(model_name="alexnet",epochs=100)
#     # recognition_ai.train(model_name="resnet",epochs=100)
#     # recognition_ai.train(model_name="efficientnet",epochs=100)
#     img = cv2.imread("Recognition/test1.png")
#     # char = recognition_ai.inference(image=img,model_name="resnet50V2--epoch48.h5")
#     char = recognition_ai.inference(image=img,model_name="alexnet--epoch25.h5",generate_csv=False)
#     print(char)
    # end = time()
    # execution_time = end - start
    # print(f"Time = {execution_time} seconds")
