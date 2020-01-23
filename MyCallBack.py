import keras
from keras.preprocessing.image import ImageDataGenerator
from plot_it import Plotter

class MyCallBack(keras.callbacks.Callback):

    def __init__(self, data, filename):
        self.data = data
        self.filename = filename
        self.plot = Plotter()

        self.datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=45,
            fill_mode='nearest',
            horizontal_flip=False,
            vertical_flip=False)

        super().__init__()

    def on_epoch_end(self, epoch, logs={}):

        Xtrain = self.data['Xtrain']
        ytrain = self.data['ytrain']
        Xtest = self.data['Xtest']
        ytest = self.data['ytest']
        Xval = self.data['Xval']
        yval = self.data['yval']

        self.datagen.fit(Xtrain)
        loss1, acc1 = self.model.evaluate_generator(self.datagen.flow(Xtrain, ytrain), steps=Xtrain.shape[0]/64)
        self.datagen.fit(Xval)
        loss2, acc2 = self.model.evaluate_generator(self.datagen.flow(Xval, yval), steps=Xval.shape[0]/64)
        self.datagen.fit(Xtest)
        loss3, acc3 = self.model.evaluate_generator(self.datagen.flow(Xtest, ytest), steps=Xtest.shape[0]/64)

        last_epoch = self.plot.find_last_epoch(self.filename + '_loss.txt')
        row = '{},{},{},{}'.format(last_epoch+1, round(loss1, 3), round(loss2, 3), round(loss3, 3))
        self.plot.write_to_files(self.filename + '_loss.txt', 'i,Train,Validation,Test', row)

        last_epoch = self.plot.find_last_epoch(self.filename + '_acc.txt')
        row = '{},{},{},{}'.format(last_epoch+1, round(acc1, 3), round(acc2, 3), round(acc3, 3))
        self.plot.write_to_files(self.filename + '_acc.txt', 'i,Train,Validation,Test', row)

