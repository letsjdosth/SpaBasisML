
import matplotlib.pyplot as plt
#FOR ML
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from test_data_generator import test_data_generator

(designMat_data, Zmat_data, gridLoc_data, designMat_cv, Zmat_cv, gridLoc_cv) = test_data_generator(dist="poisson")


# ML fit
print(designMat_data.shape) #1000, 102

inputs = keras.Input(shape=(102,), name="x_basis")#고침
x1 = layers.Dense(100, activation="relu")(inputs)
x2 = layers.Dense(50, activation="relu")(x1)
x3 = layers.Dense(20, activation="relu")(x2)
outputs = layers.Dense(1, name="predictions")(x3)
MLmodel = keras.Model(inputs=inputs, outputs=outputs)

MLmodel.summary()


MLmodel.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    metrics=["MeanSquaredError","MeanAbsoluteError"],
)

print("fit!")
history = MLmodel.fit(designMat_data, Zmat_data, batch_size=1000, epochs=700, validation_split=0.2)
print("eval! (data+cv)")
fit_scores = MLmodel.evaluate(designMat_data, Zmat_data, verbose=2)
test_scores = MLmodel.evaluate(designMat_cv, Zmat_cv, verbose=2)
ML_prediction_Y_datapt = MLmodel.predict(designMat_data)
ML_prediction_Y_cv = MLmodel.predict(designMat_cv)

# ML: figures
fig3, ((ax01, ax02), (ax03, ax04)) = plt.subplots(2,2)
ax01.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=Zmat_data)
ax01.set_title("datapoint:Obs")
ax02.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=ML_prediction_Y_datapt)
ax02.set_title("datapoint:ModelFit")
ax03.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=Zmat_cv)
ax03.set_title("crossval:Obs")
ax04.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=ML_prediction_Y_cv)
ax04.set_title("crossval:Predict")
plt.show()
