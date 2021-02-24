import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


from tensorflow.keras import backend as K

def PermaDropout(rate):
    return layers.Lambda(lambda x: K.dropout(x, level=rate))


from test_data_generator import test_data_generator





# import simulation data
n_cv = 500
(designMat_data, Zmat_data, gridLoc_data, designMat_cv, Zmat_cv, gridLoc_cv) = test_data_generator(dist="poisson", n_cv=n_cv)
X_train_covariate = np.reshape(designMat_data[:,:2], (1000,2,1))
X_train_basis = np.reshape(designMat_data[:,2:], (1000,100,1))
X_cv_covariate = np.reshape(designMat_cv[:,:2], (n_cv, 2,1))
X_cv_basis = np.reshape(designMat_cv[:,2:], (n_cv, 100,1))
print(X_cv_basis.shape)

# CNN model def
image_input = layers.Input(shape=(100,1))
conv1 = layers.Conv1D(16, 6, activation='relu')(image_input)
conv1 = layers.MaxPooling1D()(conv1)
conv1 = PermaDropout(0.2)(conv1)
conv2 = layers.Conv1D(32, 6, activation='relu')(conv1)
conv2 = layers.MaxPooling1D()(conv2)
conv2 = PermaDropout(0.2)(conv2)
first_part_output = layers.Flatten()(conv2)
merged1 = layers.Dense(12, activation='relu', name="last")(first_part_output)
out_layer = layers.Dense(1, activation='relu')(merged1)

CNN_model = keras.Model(inputs=image_input, outputs = [merged1, out_layer])
CNN_model.summary()

CNN_model.compile(optimizer='adam',
            loss='mse',
            metrics=['accuracy'])

#CNN fit
CNN_model_fit = CNN_model.fit(X_train_basis, Zmat_data, epochs=200) #input: basis only
# test_loss, test_acc = CNN_model.evaluate([X_cv_basis, X_cv_covariate], Zmat_cv, verbose=2)
# print("cv loss:", test_loss)

#CNN prediction
CNN_model_last_layer_predictions = []
sample_num = 500
for i in range(sample_num):
    CNN_model_prediction = CNN_model.predict(X_cv_basis)
    CNN_model_last_layer_predictions.append(CNN_model_prediction[0][0]) #first index: last layer idx, second idx: cv data idx
    if i%50 == 0:
        print("prediction:", i)


CNN_model_last_layer_predictions_trace = np.array(CNN_model_last_layer_predictions).T


fig, axs = plt.subplots(4, 3, sharey=True, tight_layout=True) #12 plots
for i in range(CNN_model_last_layer_predictions_trace.shape[0]):
    trace = CNN_model_last_layer_predictions_trace[i,:].flatten() #<-linter error
    axs[i//3, i%3].hist(trace, bins=50)
plt.show()
