from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, load_model
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from project_utilities import *
import scipy.io
import cv2
from keras.applications.vgg16 import VGG16
from MyCallBack import MyCallBack
from plot_it import Plotter
import numpy as np


class Cnn:

    def __init__(self, model=None, model_size=(32, 32, 3), save_dir='temp'):
        self.model = model
        self.Xtrain = None
        self.ytrain = None
        self.Xtest = None
        self.ytest = None
        self.Xval = None
        self.yval = None
        self.model_size = model_size
        self.save_dir = save_dir

        if os.path.isdir(save_dir) is False:
            os.makedirs(save_dir)

    def convolution_set(self, X, size, filter, strides, numb, padding):

        X = Conv2D(size, (filter, filter), strides=(strides, strides), name='conv_' + numb, padding=padding)(X)
        X = BatchNormalization(axis=3, name='batch_' + numb)(X)
        X = Activation('relu')(X)
        return X

    def build_model(self, load_saved_model=None, model_type='custom', opt=None):

        if load_saved_model is not None and os.path.isfile(self.save_dir + '/' + load_saved_model):
            print('Loading saved model ' + str(load_saved_model))
            self.model = load_model(self.save_dir + '/' + load_saved_model)
        elif model_type == 'custom':
            X_input = Input(self.model_size)
            depth = self.model_size[0]

            X = self.convolution_set(X_input, depth, 3, 1, '1', 'same')
            X = self.convolution_set(X, depth, 3, 1, '2', 'same')
            X = MaxPooling2D((2, 2), name='max_pool_1')(X)

            X = self.convolution_set(X, depth*2, 3, 1, '3', 'same')
            X = self.convolution_set(X, depth*2, 3, 1, '4', 'same')
            X = MaxPooling2D((2, 2), name='max_pool_2')(X)

            X = self.convolution_set(X, depth*4, 3, 1, '5', 'same')
            X = self.convolution_set(X, depth*4, 3, 1, '6', 'same')
            X = MaxPooling2D((2, 2), name='max_pool_3')(X)

            X = Flatten()(X)

            X = Dense(depth*8, activation='relu')(X)
            X = Dropout(0.25)(X)
            X = Dense(depth*8, activation='relu')(X)
            X = Dropout(0.25)(X)
            X = Dense(depth*8, activation='relu')(X)
            X = Dropout(0.25)(X)

            X = Dense(11, activation='softmax', name='y')(X)

            model = Model(inputs=X_input, outputs=X, name='my_model')
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            self.model = model

        elif model_type == 'vgg16':
            depth = self.model_size[0]
            vgg16_model = VGG16(weights='imagenet', include_top=False)

            vgg16_model.summary()

            # Used to freeze the vgg16 layers
            for layer in vgg16_model.layers:
                layer.trainable = True

            img_in = Input(shape=self.model_size, name='img_in')
            vgg16_conv = vgg16_model(img_in)
            X = Flatten()(vgg16_conv)
            X = Dense(depth*8, activation='relu')(X)
            X = Dropout(0.25)(X)
            X = Dense(depth*8, activation='relu')(X)
            X = Dropout(0.25)(X)
            X = Dense(depth*8, activation='relu')(X)
            X = Dropout(0.25)(X)
            X = Dense(11, activation='softmax', name='y')(X)

            model = Model(inputs=img_in, outputs=X, name='my_model')
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            self.model = model

        elif model_type == 'vgg16_custom':
            depth = self.model_size[0]
            vgg16_model = VGG16(weights=None, include_top=False)

            img_in = Input(shape=self.model_size, name='img_in')
            vgg16_conv = vgg16_model(img_in)

            vgg16_model.summary()

            X = Flatten()(vgg16_conv)
            X = Dense(depth*8, activation='relu')(X)
            X = Dropout(0.25)(X)
            X = Dense(depth*8, activation='relu')(X)
            X = Dropout(0.25)(X)
            X = Dense(depth*8, activation='relu')(X)
            X = Dropout(0.25)(X)
            X = Dense(11, activation='softmax', name='y')(X)

            model = Model(inputs=img_in, outputs=X, name='my_model')
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            self.model = model

        self.model.summary()

    def train_cnn(self, epochs=1, save_data='my_model', iter=0):
        if self.model is None:
                print('Empty model. Can not train!')
        else:
            print('Training model...')

            batch_size = 32

            if epochs > 0:
                #es = EarlyStopping(monitor='val_loss', patience=10)
                #tb = TensorBoard(histogram_freq=0, write_images=False)
                mc = ModelCheckpoint(self.save_dir + '/' + str(iter) + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
                lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
                call_data = {'Xtest': self.Xtest,
                             'ytest': self.ytest,
                             'Xtrain': self.Xtrain,
                             'ytrain': self.ytrain,
                             'Xval': self.Xval,
                             'yval': self.yval}

                mcb = MyCallBack(call_data, filename=self.save_dir + '/epoch_data')

                datagen_train = ImageDataGenerator(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=45,
                    fill_mode='nearest',
                    horizontal_flip=False,
                    vertical_flip=False)
                datagen_train.fit(self.Xtrain)

                datagen_val = ImageDataGenerator(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=45,
                    fill_mode='nearest',
                    horizontal_flip=False,
                    vertical_flip=False)
                datagen_val.fit(self.Xval)

                history = self.model.fit_generator(datagen_train.flow(self.Xtrain, self.ytrain, batch_size=batch_size),
                                                   steps_per_epoch=self.Xtrain.shape[0] // batch_size, epochs=epochs,
                                                   callbacks=[lr, mc, mcb],
                                                   validation_data=datagen_val.flow(self.Xval, self.yval),
                                                   validation_steps=self.Xval.shape[0] // batch_size)

                # Used to output learning curves if needed
                if False:
                    plot = Plotter()
                    plot.plot_accuracy(history, 'Model Accuracy', 'Accuracy', 'Epochs', self.save_dir + '/model_acc.png')
                    plot.plot_loss(history, 'Model Loss', 'Loss', 'Epochs', self.save_dir + '/model_loss.png')

            if save_data is not None:
                print('Saving model...')
                self.model.save(self.save_dir + '/' + save_data)

    def evaluate_model(self, eval_type='train'):

        if self.model is None:
            print('Empty model. Can not be evaluated!')
        else:
            print('Evaluating model...')
            if eval_type == 'train':
                print('Evaluating on train data...')
                X = self.Xtrain
                y = self.ytrain
            elif eval_type == 'test':
                print('Evaluating on test data...')
                X = self.Xtest
                y = self.ytest
            else:
                print('Evaluating on validation data...')
                X = self.Xval
                y = self.yval

            datagen = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=45,
                fill_mode='nearest',
                horizontal_flip=False,
                vertical_flip=False)
            datagen.fit(X)

            p = self.model.evaluate_generator(datagen.flow(X, y), steps=50)
            print("Evaluation Loss = " + str(p[0]))
            print("Evaluation Acc  = " + str(p[1]))
            print('-' * 25)

    def predict_model(self, img):
        if self.model is None:
            print('Empty model. Can not be predicted!')
        else:
            return self.model.predict(img)

    def get_data(self, subset=0, extra_train=.50, reset=False):
        print('Loading images...')

        train_file = 'train_32x32.mat'
        test_file = 'test_32x32.mat'
        val_file = 'extra_32x32.mat'
        my_size = str(self.model_size[0]) + 'x' + str(self.model_size[0])
        val_mat = None

        xtrain_pickle = 'Xtrain_' + my_size + '.pickle'
        ytrain_pickle = 'ytrain_' + my_size + '.pickle'
        xtest_pickle = 'Xtest_' + my_size + '.pickle'
        ytest_pickle = 'ytest_' + my_size + '.pickle'
        xval_pickle = 'Xval_' + my_size + '.pickle'
        yval_pickle = 'yval_' + my_size + '.pickle'

        # -------------- Train Data --------------------------------------#
        if os.path.isfile(xtrain_pickle) and os.path.isfile(ytrain_pickle) and reset is False:
            print('Loading Xtrain and ytrain from pickle files')
            with open(xtrain_pickle, 'rb') as data:
                self.Xtrain = pickle.load(data)
            with open(ytrain_pickle, 'rb') as data:
                self.ytrain = pickle.load(data)
        else:
            mat = scipy.io.loadmat(train_file)
            self.Xtrain = mat['X']
            self.Xtrain = np.transpose(self.Xtrain, (3, 0, 1, 2))
            self.ytrain = mat['y']
            self.ytrain = np.squeeze(self.ytrain.T)
            self.ytrain[self.ytrain == 10] = 0

            if self.model_size[0] != 32:
                new_imgs = []
                print('Converting {} data to: {}x{}'.format('train', self.model_size[0], self.model_size[1]))
                for i in range(self.Xtrain.shape[0]):
                    new_imgs.append(cv2.resize(self.Xtrain[i], self.model_size[:2]))
                self.Xtrain = np.asarray(new_imgs)

            val_mat = scipy.io.loadmat(val_file)
            extra_train = int(self.Xtrain.shape[0] * extra_train)
            vx = val_mat['X'][:, :, :, :extra_train]
            vx = np.transpose(vx, (3, 0, 1, 2))
            vy = val_mat['y'][:extra_train, :]
            vy = np.squeeze(vy.T)
            vy[vy == 10] = 0

            if self.model_size[0] != 32:
                new_imgs = []
                print('Converting {} data to: {}x{}'.format('extra train', self.model_size[0], self.model_size[1]))
                for i in range(vx.shape[0]):
                    new_imgs.append(cv2.resize(vx[i], self.model_size[:2]))
                vx = np.asarray(new_imgs)

            print('Loading negative train images from folder...')
            img_list = []
            windowsize = self.model_size[0]
            for sizer in [(512, 384, 16), (384, 288, 16)]:
                imgs = load_images_from_dir(size=(sizer[0], sizer[1]))
                stepsize = sizer[2]
                h, w, d = imgs[0].shape
                for img in imgs:
                    for y in range(0, h - windowsize, stepsize):
                        for x in range(0, w - windowsize, stepsize):
                            template = img[y:y + windowsize, x:x + windowsize]
                            img_list.append(template)

            img_list = np.asarray(img_list)
            self.Xtrain = np.vstack((self.Xtrain, vx, img_list))
            shuffler = np.random.permutation(self.Xtrain.shape[0])
            self.Xtrain = self.Xtrain[shuffler]
            y_list = np.ones((img_list.shape[0]), dtype=np.uint8) * 10
            self.ytrain = np.hstack((self.ytrain, vy, y_list))
            self.ytrain = self.ytrain[shuffler]
            self.ytrain = to_categorical(self.ytrain, num_classes=11)

            print('Saving: ', xtrain_pickle, ytrain_pickle)
            with open(xtrain_pickle, 'wb') as output:
                pickle.dump(self.Xtrain, output)
            with open(ytrain_pickle, 'wb') as output:
                pickle.dump(self.ytrain, output)

        # -------------- Test Data --------------------------------------#
        if os.path.isfile(xtest_pickle) and os.path.isfile(ytest_pickle) and reset is False:
            print('Loading Xtest and ytest from pickle files')
            with open(xtest_pickle, 'rb') as data:
                self.Xtest = pickle.load(data)
            with open(ytest_pickle, 'rb') as data:
                self.ytest = pickle.load(data)
        else:
            mat = scipy.io.loadmat(test_file)
            self.Xtest = mat['X']
            self.Xtest = np.transpose(self.Xtest, (3, 0, 1, 2))
            self.ytest = mat['y']
            self.ytest = np.squeeze(self.ytest.T)
            self.ytest[self.ytest == 10] = 0

            if self.model_size[0] != 32:
                new_imgs = []
                print('Converting {} data to: {}x{}'.format('test', self.model_size[0], self.model_size[1]))
                for i in range(self.Xtest.shape[0]):
                    new_imgs.append(cv2.resize(self.Xtest[i], self.model_size[:2]))
                self.Xtest = np.asarray(new_imgs)

            print('Loading negative test images from folder...')
            img_list = []
            windowsize = self.model_size[0]
            for sizer in [(512, 384, 16), (384, 288, 16)]:
                imgs = load_images_from_dir(data_dir='input_images/test', size=(sizer[0], sizer[1]))
                stepsize = sizer[2]
                h, w, d = imgs[0].shape
                for img in imgs:
                    for y in range(0, h - windowsize, stepsize):
                        for x in range(0, w - windowsize, stepsize):
                            template = img[y:y + windowsize, x:x + windowsize]
                            img_list.append(template)

            img_list = np.asarray(img_list)
            y_list = np.ones((img_list.shape[0]), dtype=np.uint8) * 10

            self.Xtest = np.vstack((self.Xtest, img_list))
            self.ytest = np.hstack((self.ytest, y_list))

            shuffler = np.random.permutation(self.Xtest.shape[0])
            self.Xtest = self.Xtest[shuffler]
            self.ytest = self.ytest[shuffler]

            self.ytest = to_categorical(self.ytest, num_classes=11)

            print('Saving: ', xtest_pickle, ytest_pickle)
            with open(xtest_pickle, 'wb') as output:
                pickle.dump(self.Xtest, output)
            with open(ytest_pickle, 'wb') as output:
                pickle.dump(self.ytest, output)

        # -------------- Validation Data --------------------------------------#
        if os.path.isfile(xval_pickle) and os.path.isfile(yval_pickle) and reset is False:
            print('Loading Xval and yval from pickle files')
            with open(xval_pickle, 'rb') as data:
                self.Xval = pickle.load(data)
            with open(yval_pickle, 'rb') as data:
                self.yval = pickle.load(data)
        else:
            if val_mat is None:
                val_mat = scipy.io.loadmat(val_file)
            self.Xval = val_mat['X'][:, :, :, val_mat['X'].shape[3]-15000:]
            self.Xval = np.transpose(self.Xval, (3, 0, 1, 2))
            self.yval = val_mat['y'][val_mat['X'].shape[3]-15000:, :]
            self.yval = np.squeeze(self.yval.T)
            self.yval[self.yval == 10] = 0

            if self.model_size[0] != 32:
                new_imgs = []
                print('Converting {} data to: {}x{}'.format('validation', self.model_size[0], self.model_size[1]))
                for i in range(self.Xval.shape[0]):
                    new_imgs.append(cv2.resize(self.Xval[i], self.model_size[:2]))
                self.Xval = np.asarray(new_imgs)

            print('Loading negative test images from folder...')
            img_list = []
            windowsize = self.model_size[0]
            for sizer in [(512, 384, 16), (384, 288, 16)]:
                imgs = load_images_from_dir(data_dir='input_images/val', size=(sizer[0], sizer[1]))
                stepsize = sizer[2]
                h, w, d = imgs[0].shape
                for img in imgs:
                    for y in range(0, h - windowsize, stepsize):
                        for x in range(0, w - windowsize, stepsize):
                            template = img[y:y + windowsize, x:x + windowsize]
                            img_list.append(template)

            img_list = np.asarray(img_list)
            self.Xval = np.vstack((self.Xval, img_list))
            shuffler = np.random.permutation(self.Xval.shape[0])
            self.Xval = self.Xval[shuffler]
            y_list = np.ones((img_list.shape[0]), dtype=np.uint8) * 10
            self.yval = np.hstack((self.yval, y_list))
            self.yval = self.yval[shuffler]
            self.yval = to_categorical(self.yval, num_classes=11)

            print('Saving: ', xval_pickle, yval_pickle)
            with open(xval_pickle, 'wb') as output:
                pickle.dump(self.Xval, output)
            with open(yval_pickle, 'wb') as output:
                pickle.dump(self.yval, output)

        print('-' * 30)
        print('Full data sets')
        print('Xtrain shape: ', self.Xtrain.shape)
        print('Xtest shape: ', self.Xtest.shape)
        print('Xval shape: ', self.Xval.shape)
        print('ytrain shape: ', self.ytrain.shape)
        print('ytest shape: ', self.ytest.shape)
        print('yval shape: ', self.yval.shape)
        print('-' * 30)

        # Used for quick iteration for debugging and showing training is working
        if subset > 0:
            shuffler = np.random.permutation(self.Xtrain.shape[0])
            shuffler = shuffler[:int(round(self.Xtrain.shape[0] * subset))]
            self.Xtrain = self.Xtrain[shuffler]
            self.ytrain = self.ytrain[shuffler]

            shuffler = np.random.permutation(self.Xtest.shape[0])
            shuffler = shuffler[:int(round(self.Xtest.shape[0] * subset))]
            self.Xtest = self.Xtest[shuffler]
            self.ytest = self.ytest[shuffler]

            shuffler = np.random.permutation(self.Xval.shape[0])
            shuffler = shuffler[:int(round(self.Xval.shape[0] * subset))]
            self.Xval = self.Xval[shuffler]
            self.yval = self.yval[shuffler]

            print('Xtrain shape: ', self.Xtrain.shape)
            print('Xtest shape: ', self.Xtest.shape)
            print('Xval shape: ', self.Xval.shape)
            print('ytrain shape: ', self.ytrain.shape)
            print('ytest shape: ', self.ytest.shape)
            print('yval shape: ', self.yval.shape)

        # Quick sanity check to see if the images and labels look to be matching up
        if False:
            for i in range(100):
                cv2.imshow('train', self.Xtrain[i].astype(np.uint8))
                print('ytrain', self.ytrain[i])
                cv2.imshow('test', self.Xtest[i].astype(np.uint8))
                print('ytest', self.ytest[i])
                cv2.imshow('val', self.Xval[i].astype(np.uint8))
                print('yval', self.yval[i])
                cv2.waitKey(0)

        # Used to create a small pickle file for training testing
        if False:
            with open('small_xtrain.pickle', 'wb') as output:
                pickle.dump(self.Xtrain[:500], output)
            with open('small_ytrain.pickle', 'wb') as output:
                pickle.dump(self.ytrain[:500], output)
            with open('small_xval.pickle', 'wb') as output:
                pickle.dump(self.Xval[:100], output)
            with open('small_yval.pickle', 'wb') as output:
                pickle.dump(self.yval[:100], output)
            with open('small_xtest.pickle', 'wb') as output:
                pickle.dump(self.Xtest[:100], output)
            with open('small_ytest.pickle', 'wb') as output:
                pickle.dump(self.ytest[:100], output)

        self.Xtrain = self.Xtrain / 255.
        self.Xval = self.Xval / 255.
        self.Xtest = self.Xtest / 255.

    def summerize_model(self):
        print(self.model.summary())
