import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


class Plotter:

    def __init__(self):
        pass

    def plot(self, df, filename='plots\default.png', title='Graph Title',
             xaxis='X Values', yaxis='Y Values', axis=(1, 50, .5, 1.)):

        ax = df.plot(title=title, fontsize=12)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        plt.grid()
        plt.legend(loc="best")
        print(axis)
        plt.axis(axis)
        plt.savefig(filename)
        #plt.show()
        plt.close()

    def write_to_files(self, filename, header, row):

        if os.path.isfile(filename) is False:
            write_header = True
        else:
            write_header = False

        f = open(filename, 'a')

        if write_header:
            f.write(header + '\n')
        f.write(row + '\n')
        f.close()

    def find_last_epoch(self, filename):

        if os.path.isfile(filename):
            with open(filename) as f:
                lines = open(filename, 'r').readlines()

            lines = [i.strip() for i in lines]
            last_epoch = int(lines[-1].split(',')[0])

            return last_epoch
        else:
            return 0

    def plot_accuracy(self, history, title, y_label, x_label, filename):

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid()
        plt.legend(['Train', 'Validation'], loc='best')
        plt.savefig(filename)
        #plt.show()
        plt.close()

    def plot_loss(self, history, title, y_label, x_label, filename):

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid()
        plt.legend(['Train', 'Validation'], loc='best')
        plt.savefig(filename)
        #plt.show()
        plt.close()


if __name__ == '__main__':
    my_plot = Plotter()

    my_plot.write_to_files('my_plot_test.txt', 'i,a,b,c,d', '1,2,3,4,5')

    print(my_plot.find_last_epch('train_val_test.txt'))

    df = pd.read_csv('train_val_test.txt', index_col=0)
    my_plot.plot(df, 'plots/test.png', 'Test Title', 'Epoch', 'Accuracy')
