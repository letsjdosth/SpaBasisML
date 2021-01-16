import numpy as np
import matplotlib.pyplot as plt


def show_vavlidation_plot(gridLoc, true, predicted):
    """
    gridLoc : numpy ndarray [[x1,y1],[x2,y2],...,[x_n,y_n]] 
    true : numpy ndarray n by 1
    predicted : numpy ndarray n by 1
    """
    data_num = gridLoc.shape[0]
    true = np.reshape(true,(data_num,))
    predicted = np.reshape(predicted,(data_num,))

    fig, ((ax01, ax02)) = plt.subplots(1,2)
    ax01.scatter(gridLoc[:,0], gridLoc[:,1], s=1, c=(true - min(true))/(max(true) - min(true)), cmap="Reds")
    ax01.set_title("Obs:Validation")
    ax02.scatter(gridLoc[:,0], gridLoc[:,1], s=1, c=, cmap="Reds")
    ax02.set_title("predicted")
    plt.show()
