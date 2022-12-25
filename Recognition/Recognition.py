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


# dataset_path = os.path.join("Recognition","English","Fnt")
# model_path = os.path.join("Recognition","Model")

# Set GPU device
# cv2.cuda.setDevice(0)

class RECOGNITION:
    model_path = os.path.join("Recognition","Model")
    dataset_path = os.path.join("Recognition","Chars-Dataset","Fnt")
    checkpoint_path = os.path.join("Recognition","Model","Checkpoints")
    logs_paths = os.path.join("Recognition","Model","Logs")
    width = 128
    height = 128
    depth = 3
    
    def alexnet_model(self,num_classes, input_shape):
        model  = Sequential()
        # 1st conv layer
        model.add(Conv2D(filters=96, input_shape = input_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
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
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        return model 
    
    def train(self):
        directory = self.dataset_path
        batch_size = 128
        epochs = 50
        width = self.width
        height = self.height
        depth = self.depth
        
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
        
        with tf.device('/device:GPU:0'):
            # Setup Callbacks for Checkpoint epochs
            checkpoint_file = os.path.join(self.checkpoint_path,"model--epoch{epoch:02d}.h5")
            # checkpoint_file = os.path.join(self.checkpoint_path,"model--{epoch:02d}--{val_acc:.2f}.h5")
            
            save_callback = ModelCheckpoint(
                filepath=checkpoint_file,
                monitor= 'val_accuracy',#'val_loss', 
                verbose=1,
                save_best_only=False,
                mode='auto'
            )
            
            # Setup callback to reduce learning rate as epochs progress
            reduce_lr = ReduceLROnPlateau(
                monitor='loss', 
                factor=0.2,
                patience=3, 
                min_lr=0.001
            )
            
            # Setup callback to stop training if loss doesn't decrease
            stop_callback = EarlyStopping(
                monitor='val_loss',
                verbose=1,
                patience=30
            )
            
            # Setup for TensorBoard Visualizations
            tensorboard = TensorBoard(log_dir=self.logs_paths)
            
            # Setup CSV Logger to log every epoch to a csv file
            csv_callback = CSVLogger(
                filename=os.path.join(self.model_path,"training_logs.csv"),
                separator=",",
                append=False
            )
            
            callback_list = [reduce_lr, save_callback, tensorboard, csv_callback, stop_callback]
            
            model = self.alexnet_model(num_classes,input_shape)
            
            model.compile(
                loss = 'categorical_crossentropy', 
                optimizer = 'adam', 
                metrics = ['accuracy']
            )
            model.summary()
            
            model.fit(
                train_generator, 
                steps_per_epoch= train_generator.n//train_generator.batch_size, 
                epochs=epochs,
                callbacks=callback_list,
                validation_data = validation_generator, 
                validation_steps = validation_generator.n//validation_generator.batch_size)

    # def convert2ONNX(self):
    #     model_path = os.path.join(self.model_path,"model.h5")
    #     model = load_model(model_path)
    #     onnx_model = onnxmltools.convert_keras(model)
    #     onnx_path = os.path.join(self.model_path,"model.onnx")
    #     onnxmltools.save_model(onnx_model,onnx_path)
        
    def infer(self,image):
        width = 128
        height = 128
        chars = [
        '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H',
        'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
        ]
        
        img_src = cv2.imread(image)  
        src = cv2.resize(img_src,(width,height))
        src = src.astype('float')/255.0
        src = img_to_array(src)
        src = np.expand_dims(src, axis=0)         
        
        with tf.device('/device:GPU:0'):
            model = load_model(os.path.join(self.model_path,"model.h5"), compile=False)
            predictions = model.predict(src)
    
            predictions_df = pd.DataFrame(predictions)
            predictions_df = predictions_df.T 
            predictions_df = predictions_df.rename(columns={0:"Confidence"})    
            predictions_df = predictions_df.assign(Character=chars)
            predictions_df["Confidence"] = predictions_df["Confidence"].multiply(100)
            
            
            top_predictions = predictions_df.nlargest(n=3, columns="Confidence")
            top_predictions.to_csv(os.path.join("Recognition","test.csv"))
            # Select the top-3 predictions with the highest confidence for each sample
            # top_k = 3
            # predictions_top_k = predictions_df.apply(lambda x: x.nlargest(top_k).index.tolist(), axis=1)
            # print(predictions_top_k)
            # confidences_top_k = predictions_df.apply(lambda x: x.nlargest(top_k).values, axis=1)
            
            # new_df = pd.DataFrame(confidences_top_k.values,columns=predictions_top_k.values)
            
            # new_df.to_csv(os.path.join("Recognition","test.csv"))
            # print(predictions_top_k)
            
            # idx = np.argsort(predictions)[-1]
        # return chars[idx]
            

        

        
        
        # cv2.imshow("image",source)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    recognition_ai = RECOGNITION()
    # recognition_ai.convert2ONNX()
    recognition_ai.infer(image="Recognition/test1.png")

