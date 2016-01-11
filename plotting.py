import matplotlib as mpl
mpl.use('TkAgg')

from matplotlib import pyplot as plt
import numpy as np

FIGURE_TITLE_FONT_SIZE = 16

def allow_plot():
    plt.ion()


class LossPlot(object):

    def __init__(self, fignum):
        self._fig = plt.figure(fignum)
        self._fig.suptitle('Loss', fontsize=FIGURE_TITLE_FONT_SIZE)
        plt.grid()
        self._train_loss_plot, = plt.plot([], [], label='Training Loss')
        self._valid_loss_plot, = plt.plot([], [], label='Validation Loss')
        plt.legend()
        self._ax = plt.gca()

    def plot(self, train_loss, valid_loss, i_epoch):
        self._train_loss_plot.set_xdata(np.append(self._train_loss_plot.get_xdata(), i_epoch))
        self._train_loss_plot.set_ydata(np.append(self._train_loss_plot.get_ydata(), train_loss))

        self._valid_loss_plot.set_xdata(np.append(self._valid_loss_plot.get_xdata(), i_epoch))
        self._valid_loss_plot.set_ydata(np.append(self._valid_loss_plot.get_ydata(), valid_loss))
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw()


class AccuracyPlot(object):

    def __init__(self, fignum):
        self._fig = plt.figure(fignum)
        self._fig.suptitle('Accuracy', fontsize=FIGURE_TITLE_FONT_SIZE)
        plt.grid()
        self._train_acc_plot, = plt.plot([], [], label='Training Accuracy')
        self._valid_acc_plot, = plt.plot([], [], label='Validation Accuracy')
        plt.legend()
        self._ax = plt.gca()

    def plot(self, train_acc, valid_acc, i_epoch):
        self._train_acc_plot.set_xdata(np.append(self._train_acc_plot.get_xdata(), i_epoch))
        self._train_acc_plot.set_ydata(np.append(self._train_acc_plot.get_ydata(), train_acc))

        self._valid_acc_plot.set_xdata(np.append(self._valid_acc_plot.get_xdata(), i_epoch))
        self._valid_acc_plot.set_ydata(np.append(self._valid_acc_plot.get_ydata(), valid_acc))
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw()