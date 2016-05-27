from util import Progplot
import numpy as np

n_epoch = 30

progplot = Progplot(n_epoch, "epochs",
                    names=['loss (train)', 'loss (val.)'])

loss_train = np.random.randn(n_epoch)
loss_val = np.random.randn(n_epoch)


for i in range(n_epoch):
    progplot.add(values=[("loss (train)", loss_train[i]),
                         ("loss (val.)", loss_val[i])])
