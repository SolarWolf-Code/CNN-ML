# create a cnn with google inception
from keras.applications.inception_v3 import InceptionV3
# Import libraries
from cgi import test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#to plot accuracy
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split #to split training and testing data
from keras.utils import to_categorical#to convert the labels present in y_train and t_test into one-hot encoding
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout#to create CNN
import os
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Input
# import image sharpening library
from PIL import ImageFilter
from PIL import Image, ImageEnhance
import random
# import globalaveragepooling2d
from keras.layers import GlobalAveragePooling2D
import time
tf.config.run_functions_eagerly(False)

def train_data_aug(image):
                changes = ["contrast", "brightness", "rotation", "sharpness"]
                choice = random.choice(changes)
                if choice == "contrast":
                        enhancer = ImageEnhance.Contrast(image)
                        factor = random.uniform(0.5, 1.5)
                        image = enhancer.enhance(factor)
                elif choice == "brightness":
                        enhancer = ImageEnhance.Brightness(image)
                        factor = random.uniform(0.5, 1.5)
                        image = enhancer.enhance(factor)
                elif choice == "rotation":
                        image = image.rotate(random.randint(0, 360))
                        # show image next to the original
                elif choice == "sharpness":
                        image = image.filter(ImageFilter.SHARPEN)

                return image

def data_creation():
        data = []
        labels = []
        classes = 43
        cur_path = 'C:\\Users\\brads\\OneDrive\\Desktop\\BYUI\\Programming\\case_study4\\'
        #Retrieving the images and their labels
        for i in range(classes):
                if i < 10:
                        path = os.path.join(cur_path,'training',str(f"0000{i}"))
                else:
                        path = os.path.join(cur_path,'training',str(f"000{i}"))
                images = os.listdir(path)
                for a in images:
                        image = Image.open(path + "\\" + a)
                        # apply a random change to the image
                        image = train_data_aug(image)

                        image = image.resize((75,75))
                        # greyscale image
                        image = np.array(image)

                        data.append(image)
                        labels.append(i)

        #Converting lists into numpy arrays
        data = np.array(data)
        labels = np.array(labels)
        #Splitting training and testing dataset
        X_t1, X_t2, y_t1, y_t2 = train_test_split(data, labels, test_size=0.2)
        # reshape the data
        X_t1 = tf.expand_dims(X_t1, axis=-1)
        X_t2 = tf.expand_dims(X_t2, axis=-1)

        #Converting the labels into one hot encoding
        y_t1 = to_categorical(y_t1, 43)
        y_t2 = to_categorical(y_t2, 43)

        return X_t1, X_t2, y_t1, y_t2


count = 0
def main():
        global count
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import random
        import time
        
        with tf.device('/gpu:0'):
                # create the inception model
                model = InceptionV3(include_top=True, input_shape=(75, 75, 3), pooling='max', classes=43, classifier_activation='softmax', weights=None)
                # create the model
                # compile the model
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                # train the model
                while True:
                        print(f"Starting model #{count}...")
                        # start timer
                        start = time.time()
                        X_t1, X_t2, y_t1, y_t2 = data_creation()


                        tf.keras.backend.clear_session()
                        model.fit(X_t1, y_t1, epochs=20, batch_size=256, validation_data=(X_t2, y_t2))

                        # save the model
                        

                        # load model
                        #model = load_model(f'C:\\Users\\brads\\OneDrive\\Desktop\\BYUI\\Programming\\case_study4\\[98.50]inception_model.h5')

                        # evaluate the model
                        scores = model.evaluate(X_t2, y_t2)
                        # make accuracy of model a variable
                        vali_accuracy = scores[1]*100
                        # print the accuracy
                        print(f'VALIDATION accuracy: {vali_accuracy:.2f}%')

                        

                        y_test = pd.read_csv('BYUI\\Programming\\case_study4\\test_classes_partial.csv')
                        # first value is the image name, second value is the class. Separate into two lists, imgs and labels
                        imgs = y_test.iloc[:,0]
                        labels = y_test.iloc[:,1]
                        test_path = 'C:\\Users\\brads\\OneDrive\\Desktop\\BYUI\\Programming\\case_study4\\test_partial'

                        data=[]
                        for img in imgs:
                                image = Image.open(test_path + "\\" + img)
                                image = image.resize((75,75))

                                data.append(np.array(image))
                        X_test=np.array(data)
                        # predict the label
                        predict_x=model.predict(X_test) 
                        # get accuracy
                        accuracy = np.mean(np.argmax(predict_x, axis=1) == labels) * 100
                        print("TEST accuracy: %.2f%%" % (accuracy))

                        average_acc = (vali_accuracy + accuracy)/2



                        files = os.listdir(f'C:\\Users\\brads\\OneDrive\\Desktop\\BYUI\\Programming\\case_study4\\inception_models\\')


                        if len(files) == 0:
                                # if there are no files in the directory, then set the accuracy to the current accuracy
                                model.save(f'C:\\Users\\brads\\OneDrive\\Desktop\\BYUI\\Programming\\case_study4\\inception_models\\={count}=[{average_acc}]inception_model.h5')


                        else:
                                # get all files in the directory
                                files = os.listdir(f'C:\\Users\\brads\\OneDrive\\Desktop\\BYUI\\Programming\\case_study4\\inception_models\\')
                                # get the last file in the directory
                                last_file = files[0]
                                # get the last file name
                                last_file_acc = last_file.split('[')[1].split(']')[0]
                                if float(last_file_acc) < average_acc:
                                        print("Found better model!")
                                        # delete the last file
                                        os.remove(f'C:\\Users\\brads\\OneDrive\\Desktop\\BYUI\\Programming\\case_study4\\inception_models\\{last_file}')
                                        model.save(f'C:\\Users\\brads\\OneDrive\\Desktop\\BYUI\\Programming\\case_study4\\inception_models\\={count}=[{average_acc}]inception_model.h5')

                        

                        count += 1
                        

                        #model.save(f'C:\\Users\\brads\\OneDrive\\Desktop\\BYUI\\Programming\\case_study4\\[{accuracy}]inception_model.h5')

                        # end timer
                        end = time.time()
                        print(f"Time taken: {end - start}")


main()