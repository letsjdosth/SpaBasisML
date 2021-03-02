import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


from tensorflow.keras import backend as K

tf.random.set_seed(2021)

def PermaDropout(rate):
    return layers.Lambda(lambda x: K.dropout(x, level=rate))


from dataset_NKI import Dataset as NKI_Dataset


# import NKI data
NKI_data = NKI_Dataset()
print(NKI_data.data['train_x'].shape) #102,268,268
X_train_covariate = NKI_data.data['train_cov']
X_train_basis = NKI_data.data['train_x']
Zmat_data = NKI_data.data['train_y']

# print(X_train_basis)
# print(Zmat_data)


# CNN model def
image_input = layers.Input(shape=(268,268,1))
conv1 = layers.Conv2D(8, (2,2), activation='sigmoid')(image_input)
conv1 = PermaDropout(0.2)(conv1)
conv1 = layers.MaxPooling2D()(conv1)

conv2 = layers.Conv2D(4, (2,2), activation='sigmoid')(conv1)
conv2 = PermaDropout(0.2)(conv2)
conv2 = layers.MaxPooling2D()(conv2)

first_part_output = layers.Flatten()(conv2)
merged1 = layers.Dense(6, activation='relu', name="last")(first_part_output)
out_layer = layers.Dense(1, activation='relu')(merged1)

CNN_model = keras.Model(inputs=image_input, outputs = [merged1, out_layer])
CNN_model.summary()

CNN_model.compile(optimizer='adam',
            loss='mse',
            metrics=['accuracy'])

# CNN fit
CNN_model_fit = CNN_model.fit(X_train_basis, Zmat_data, epochs=20) #input: basis only
# test_loss, test_acc = CNN_model.evaluate([X_cv_basis, X_cv_covariate], Zmat_cv, verbose=2)
# print("cv loss:", test_loss)

# CNN prediction
CNN_model_last_layer_predictions = []


sample_num = 100
for i in range(sample_num):
    CNN_model_prediction = CNN_model.predict(X_train_basis)
    for j in range(102):
        CNN_model_last_layer_predictions.append(CNN_model_prediction[0][j]) #first index: last layer idx, second idx: cv data idx
    
    if i%50 == 0:
        print("prediction:", i)


CNN_model_last_layer_predictions_trace = np.array(CNN_model_last_layer_predictions).T
# print(CNN_model_last_layer_predictions_trace[0,:])
# print(CNN_model_last_layer_predictions_trace[1,:])
# print(CNN_model_last_layer_predictions_trace[2,:])
# print(CNN_model_last_layer_predictions_trace[3,:])
# print(CNN_model_last_layer_predictions_trace[4,:])
# print(CNN_model_last_layer_predictions_trace[5,:])


fig, axs = plt.subplots(2, 3, sharey=False, tight_layout=True) #12 plots
for i in range(CNN_model_last_layer_predictions_trace.shape[0]):
    trace = CNN_model_last_layer_predictions_trace[i,:].flatten() #<-linter error
    axs[i//3, i%3].hist(trace, bins=50)
plt.show()
