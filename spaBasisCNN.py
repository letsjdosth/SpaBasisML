#STD
import math

#scipy family
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

# #FOR ML
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from test_data_generator import test_data_generator

#########################################################
# main


def to_gray_image(input_designMat, covariate_dim, outputimage_2d_dim=(10,10)):
    return_gray_image_list = []
    for i in range(len(input_designMat[:,covariate_dim:])):
        now_image = input_designMat[i,covariate_dim:]
        now_image = now_image.reshape(outputimage_2d_dim)
        return_gray_image_list.append(now_image)
    return return_gray_image_list


#RGB converting (linearly)
def from_gray_to_rgb_image_Thinplate(input_gray_image_list):
    #thinplate obs max-min
    # print(np.max(designMat_data[:,2:]), np.min(designMat_data[:,2:]))
    # 0.6238621017222477 -0.18393972058459956
    #thinplate theoretical max-min
    # print(max_TPSval, min_TPSval, range_TPSval)
    # 0.6931471805599455 -0.18393972058572117 0.8770869011456667
    max_TPSval = (math.sqrt(2)**2) * (np.log(math.sqrt(2)))
    min_TPSval = np.exp(-1) * (-0.5)
    range_TPSval = max_TPSval - min_TPSval

    return_rgb_image_list = []
    for gray_image in input_gray_image_list:
        # red_mat = (gray_image - min_TPSval) / range_TPSval
        # blue_mat = (1 - ((gray_image - min_TPSval) / range_TPSval))
        # green_mat = abs(0.5-((gray_image - min_TPSval) / range_TPSval))

        
        blue_mat = np.exp(-(gray_image - min_TPSval) / range_TPSval)
        red_mat = 1-blue_mat/2
        green_mat = np.zeros((10,10))

        rgb_image = np.dstack((red_mat, green_mat, blue_mat))
        return_rgb_image_list.append(rgb_image)
        # print(return_rgb_image_list[0].shape) #(10, 10, 3)
    return return_rgb_image_list

if __name__ == "__main__":
    (designMat_data, Zmat_data, gridLoc_data, designMat_cv, Zmat_cv, gridLoc_cv) = test_data_generator(dist="poisson")

    gray_image_list_data = to_gray_image(designMat_data, 2, (10,10))
    gray_image_list_cv = to_gray_image(designMat_cv, 2, (10,10))


    # # gray image plot
    # fig1, ((ax101, ax102), (ax103, ax104)) = plt.subplots(2,2)
    # print(gray_image_list_data[0])
    # ax101.matshow(gray_image_list_data[0], cmap='gray')
    # ax102.matshow(gray_image_list_data[1], cmap='gray')
    # ax103.matshow(gray_image_list_data[2], cmap='gray')
    # ax104.matshow(gray_image_list_data[3], cmap='gray')
    # plt.show()



    rgb_image_list_data = from_gray_to_rgb_image_Thinplate(gray_image_list_data)
    rgb_image_list_cv = from_gray_to_rgb_image_Thinplate(gray_image_list_cv)



    # #rgb image plot
    # for i in range(0,30,10):
    #     fig2, ((ax201, ax202, ax203, ax204, ax205), (ax206, ax207, ax208, ax209, ax210)) = plt.subplots(2,5)
    #     ax201.imshow(rgb_image_list_data[i])
    #     ax202.imshow(rgb_image_list_data[i+1])
    #     ax203.imshow(rgb_image_list_data[i+2])
    #     ax204.imshow(rgb_image_list_data[i+3])
    #     ax205.imshow(rgb_image_list_data[i+4])
    #     ax206.imshow(rgb_image_list_data[i+5])
    #     ax207.imshow(rgb_image_list_data[i+6])
    #     ax208.imshow(rgb_image_list_data[i+7])
    #     ax209.imshow(rgb_image_list_data[i+8])
    #     ax210.imshow(rgb_image_list_data[i+9])
    #     plt.show()
    

#############################################################################################


    ## 1d version(gray)
    # (designMat_data, Zmat_data, gridLoc_data, designMat_cv, Zmat_cv, gridLoc_cv)

    X_train_covariate = np.reshape(designMat_data[:,:2], (1000,2,1))
    X_train_basis = np.reshape(designMat_data[:,2:], (1000,100,1))
    X_cv_covariate = np.reshape(designMat_cv[:,:2], (500,2,1))
    X_cv_basis = np.reshape(designMat_cv[:,2:], (500,100,1))

    print(np.max(X_train_covariate), np.min(X_train_covariate))

    image_input = layers.Input(shape=(100,1))
    other_input = layers.Input(shape=(2,))
    
    conv1 = layers.Conv1D(16, 6, activation='relu')(image_input)
    conv1 = layers.MaxPooling1D()(conv1)
    conv1 = layers.Dropout(0.2)(conv1)
    conv2 = layers.Conv1D(32, 6, activation='relu')(conv1)
    conv2 = layers.MaxPooling1D()(conv2)
    conv2 = layers.Dropout(0.2)(conv2)
    first_part_output = layers.Flatten()(conv2)
    merged_model = layers.concatenate([first_part_output, other_input])
    merged1 = layers.Dense(512, activation='relu')(merged_model)
    last_layer = layers.Dense(2, activation='relu')(merged1)

    class DeepEnsemblesLossFunction(keras.losses.Loss):
        def call(self, y_true, y_pred):
            y_true = keras.backend.cast(y_true, 'float32') #m*1
            y_pred = keras.backend.cast(y_pred, 'float32') #m*2
            y_pred_mu = y_pred[:,0]
            y_pred_sigma2 = y_pred[:,1] + keras.backend.epsilon()
            return keras.backend.sum(keras.backend.log(y_pred_sigma2)/2 + keras.backend.square(y_true - y_pred_mu) /(2*y_pred_sigma2))

    CNN_model = keras.Model(inputs=[image_input, other_input], outputs = last_layer)
    CNN_model.summary()

    CNN_model.compile(optimizer='adam',
              loss=DeepEnsemblesLossFunction(),
            #   loss='mse',
              metrics=['accuracy'])

    model_fit = CNN_model.fit([X_train_basis, X_train_covariate], Zmat_data, epochs=300)
    
    test_loss, test_acc = CNN_model.evaluate([X_cv_basis, X_cv_covariate], Zmat_cv, verbose=2)
    print("cv loss:", test_loss)


    CNN_prediction_Y_data_pt = CNN_model.predict([X_train_basis, X_train_covariate])
    CNN_prediction_mu_data_pt = CNN_prediction_Y_data_pt[:,0]
    CNN_prediction_sigma2_data_pt = CNN_prediction_Y_data_pt[:,1]

    CNN_prediction_Y_cv = CNN_model.predict([X_cv_basis, X_cv_covariate])
    CNN_prediction_mu_cv = CNN_prediction_Y_cv[:,0]
    CNN_prediction_sigma2_cv = CNN_prediction_Y_cv[:,1]
    
    print(CNN_prediction_Y_cv[0:5,:])

    # ML: figures
    fig3, ((ax01, ax02), (ax03, ax04)) = plt.subplots(2,2)
    ax01.scatter(gridLoc_data[:,0], gridLoc_data[:,1], s=Zmat_data, c=Zmat_data/max(Zmat_data), cmap="Reds")
    ax01.set_title("datapoint:Obs")
    ax02.scatter(gridLoc_data[:,0], gridLoc_data[:,1], s=CNN_prediction_mu_data_pt, c=np.reshape(CNN_prediction_mu_data_pt,(1000,)), cmap="Reds")
    ax02.set_title("datapoint:ModelFit")
    
    ax03p = ax03.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], s=Zmat_cv, c=Zmat_cv/max(Zmat_cv), cmap="Reds")
    ax03.set_title("crossval:Obs")
    ax04.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], s=CNN_prediction_mu_cv, c=np.reshape(CNN_prediction_mu_cv,(500,)), cmap="Reds")
    ax04.set_title("crossval:Predict")
    plt.show()




    #####################################################################################
    # # 2d version(gray)
    # X_train_basis = np.reshape(np.array(gray_image_list_data), (1000, 10, 10, 1))
    # X_cv_basis = np.reshape(np.array(gray_image_list_cv), (500, 10, 10, 1))
    # X_train_covariate = np.reshape(designMat_data[:,:2], (1000,2,1))
    # X_cv_covariate = np.reshape(designMat_cv[:,:2], (500,2,1))

    # image_input = layers.Input(shape=(10,10,1))
    # other_input = layers.Input(shape=(2,))
    
    # conv1 = layers.Conv2D(24, (2,2), activation='relu')(image_input)
    # conv1 = layers.MaxPooling2D((2,2))(conv1)
    # conv2 = layers.Conv2D(48, (2,2), activation='relu')(conv1)
    # conv2 = layers.MaxPooling2D((2,2))(conv2)
    # first_part_output = layers.Flatten()(conv2)
    # merged_model = layers.concatenate([first_part_output, other_input])
    # merged1 = layers.Dense(512, activation='relu')(merged_model)
    # last_layer = layers.Dense(1, activation='relu')(merged1)

    # CNN_model = keras.Model(inputs=[image_input, other_input], outputs = last_layer)
    # CNN_model.summary()

    # CNN_model.compile(optimizer='adam',
    #           loss='mse',
    #           metrics=['accuracy'])

    # model_fit = CNN_model.fit([X_train_basis, X_train_covariate], Zmat_data, epochs=300)
    
    # test_loss, test_acc = CNN_model.evaluate([X_cv_basis, X_cv_covariate], Zmat_cv, verbose=2)
    # print("cv loss:", test_loss)


    # CNN_prediction_Y_datapt = CNN_model.predict([X_train_basis, X_train_covariate])
    # CNN_prediction_Y_cv = CNN_model.predict([X_cv_basis, X_cv_covariate])

    # # ML: figures
    # fig3, ((ax01, ax02), (ax03, ax04)) = plt.subplots(2,2)
    # ax01.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=Zmat_data)
    # ax01.set_title("datapoint:Obs")
    # ax02.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=CNN_prediction_Y_datapt)
    # ax02.set_title("datapoint:ModelFit")
    # ax03.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=Zmat_cv)
    # ax03.set_title("crossval:Obs")
    # ax04.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=CNN_prediction_Y_cv)
    # ax04.set_title("crossval:Predict")
    # plt.show()




# ==========================================================================
## RGB 2d version
    # X_train_basis = np.reshape(np.array(rgb_image_list_data), (1000, 10, 10, 3))
    # X_cv_basis = np.reshape(np.array(rgb_image_list_cv), (500, 10, 10, 3))
    # X_train_covariate = np.reshape(designMat_data[:,:2], (1000,2,1))
    # X_cv_covariate = np.reshape(designMat_cv[:,:2], (500,2,1))

    # image_input = layers.Input(shape=(10,10,3))
    # other_input = layers.Input(shape=(2,))
    
    # conv1 = layers.Conv2D(16, (2,2), activation='relu')(image_input)
    # conv1 = layers.MaxPooling2D((2,2))(conv1)
    # conv2 = layers.Conv2D(32, (2,2), activation='relu')(conv1)
    # conv2 = layers.MaxPooling2D((2,2))(conv2)
    # first_part_output = layers.Flatten()(conv2)
    # merged_model = layers.concatenate([first_part_output, other_input])
    # merged1 = layers.Dense(512, activation='relu')(merged_model)
    # last_layer = layers.Dense(1, activation='relu')(merged1)

    # CNN_model = keras.Model(inputs=[image_input, other_input], outputs = last_layer)
    # CNN_model.summary()

    # CNN_model.compile(optimizer='adam',
    #           loss='mse',
    #           metrics=['accuracy'])

    # model_fit = CNN_model.fit([X_train_basis, X_train_covariate], Zmat_data, epochs=300)
    
    # test_loss, test_acc = CNN_model.evaluate([X_cv_basis, X_cv_covariate], Zmat_cv, verbose=2)
    # print("cv loss:", test_loss)


    # CNN_prediction_Y_datapt = CNN_model.predict([X_train_basis, X_train_covariate])
    # CNN_prediction_Y_cv = CNN_model.predict([X_cv_basis, X_cv_covariate])

    # # ML: figures
    # fig3, ((ax01, ax02), (ax03, ax04)) = plt.subplots(2,2)
    # ax01.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=Zmat_data)
    # ax01.set_title("datapoint:Obs")
    # ax02.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=CNN_prediction_Y_datapt)
    # ax02.set_title("datapoint:ModelFit")
    # ax03.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=Zmat_cv)
    # ax03.set_title("crossval:Obs")
    # ax04.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=CNN_prediction_Y_cv)
    # ax04.set_title("crossval:Predict")
    # plt.show()

