#STD
import math

#scipy family
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

#FOR GLM
import statsmodels.api as sm

#FOR ML
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


n_knot = 100 # set n**2 form, where n is natural number
n_data = 1000
n_cv = math.floor(n_data*0.5)

# set the places
randGenerator = np.random.default_rng(seed=12345)
gridLoc_knot = np.array([(x, y) for x in range(math.floor(n_knot**0.5)) for y in range(math.floor(n_knot**0.5))], dtype=float)/math.floor(n_knot**0.5)
gridLoc_data = randGenerator.random((n_data,2))
gridLoc_cv = randGenerator.random((n_cv,2))
gridLoc_complete = np.concatenate((gridLoc_knot, gridLoc_data, gridLoc_cv), axis=0)

# spatial covariate
Xmat_knot = randGenerator.random((n_knot,2))  #2 dim covariate
Xmat_data = randGenerator.random((n_data,2))
Xmat_cv = randGenerator.random((n_cv,2))
Xmat_complete = np.concatenate((Xmat_knot, Xmat_data, Xmat_cv), axis=0)
# print(Xmat_complete.shape) #(1600,2)

# distance matrix
distMat_complete = cdist(gridLoc_complete, gridLoc_complete,'euclidean')
# print(distMat_complete.shape) #(1600,1600)

# Generate Data following model
MaternCov_params = {"phi" : 0.2, "sigma2" : 1, "beta" : np.array([2,2], dtype=float)}
MaternCov_mat = (1 + (5**0.5 * distMat_complete / MaternCov_params["phi"]) +(5 * distMat_complete**2) / (3*MaternCov_params["phi"]**2)) * \
    np.exp(-(5**0.5 *(distMat_complete/MaternCov_params["phi"])))
# print(MaternCov_mat.shape) #(1600,1600)
MaternCov_mat *= MaternCov_params["sigma2"]
MaternCov_TSseed = randGenerator.normal(loc=0.0, scale=1.0, size=n_knot + n_data + n_cv)
Wmat = np.matmul(cholesky(MaternCov_mat).T, MaternCov_TSseed)
# print(Wmat.shape) #(1600,)
# print(np.max(Wmat), np.min(Wmat))
Zmat_intensity = np.exp(Wmat + np.matmul(Xmat_complete, MaternCov_params['beta']))
Zmat = randGenerator.poisson(lam=Zmat_intensity)
# print(Zmat.size) #1600

# thin plate splinte basis
# knot data CV
ind_knot = slice(n_knot)
ind_data = slice(n_knot, n_data+n_knot)
ind_cv = slice(n_knot+n_data, n_knot+n_data+n_cv)
distMat_data_knot = distMat_complete[ind_data, ind_knot]
distMat_cv_knot = distMat_complete[ind_cv, ind_knot]
TPSmat_data = (distMat_data_knot**2) * (np.log(distMat_data_knot))
TPSmat_cv = (distMat_cv_knot**2) * (np.log(distMat_cv_knot))
# print(TPSmat_data.shape) #1000, 100
# print(TPSmat_cv.shape) #500, 100

# design matrix
designMat_data = np.c_[Xmat_data, TPSmat_data] #여기에 1을 넣어버리면 인생이 좀 더 편해질듯
designMat_cv = np.c_[Xmat_cv, TPSmat_cv]
# print(designMat_data.shape) #1000, 102
# print(designMat_cv.shape) #500, 102



# # EDA: figures
# fig, (ax1, ax2, ax3) = plt.subplots(1,3)
# ax1.scatter(gridLoc_complete[:,0], gridLoc_complete[:,1], c="blue", s=(Wmat+abs(np.min(Wmat)))*3)
# ax1.set_title("RE(Wmat)")
# ax2.scatter(gridLoc_complete[:,0], gridLoc_complete[:,1], c="blue", s=Zmat_intensity)
# ax2.set_title("Intensity:Expected Obs")
# ax3.scatter(gridLoc_complete[:,0], gridLoc_complete[:,1], c="blue", s=Zmat)
# ax3.set_title("Obs")
# plt.show()


# #glm fit
# glm_pois = sm.GLM(Zmat[ind_data], np.c_[np.ones(n_data), designMat_data], family=sm.families.Poisson())
# glm_result = glm_pois.fit()
# print(glm_result.summary())
# # print(glm_result.params)
# glm_param = np.array(glm_result.params, dtype=float)
# glm_TPSparam = np.array(glm_result.params[3:], dtype=float)
# # print(glm_TPSparam)


# # GLM: plot the fit result
# fig2, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(2,4)
# ax11.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=Wmat[ind_data])
# ax11.set_title("Wmat:data")

# ax12.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=Wmat[ind_cv])
# ax12.set_title("Wmat:cv")

# ax13.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=Zmat[ind_data])
# ax13.set_title("Zmat:data")

# ax14.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=Zmat[ind_cv])
# ax14.set_title("Zmat:cv")

# estimatedWmat_data = np.matmul(TPSmat_data, glm_TPSparam)
# ax21.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=estimatedWmat_data)
# ax21.set_title("FIT:Wmat:data")

# estimatedWmat_cv = np.matmul(TPSmat_cv, glm_TPSparam)
# ax22.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=estimatedWmat_cv)
# ax22.set_title("EST:Wmat:cv")

# estimatedZmat_data = np.exp(np.matmul(np.c_[np.ones(n_data), designMat_data], glm_param))
# ax23.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=estimatedZmat_data)
# ax23.set_title("FIT:Zmat:data")

# estimatedZmat_cv = np.exp(np.matmul(np.c_[np.ones(n_cv), designMat_cv], glm_param))
# ax24.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=estimatedZmat_cv)
# ax24.set_title("EST:Zmat:cv")

# plt.show()

# # MSE
# glmMSE_data = sum((estimatedZmat_data - Zmat[ind_data])**2)/n_data
# glmMSE_CV = sum((estimatedZmat_cv - Zmat[ind_cv])**2)/n_cv


# ML fit
print(designMat_data.shape) #1000, 102

inputs = keras.Input(shape=(102,), name="x_basis")
x1 = layers.Dense(100, activation="relu")(inputs)
x2 = layers.Dense(50, activation="relu")(x1)
x3 = layers.Dense(20, activation="relu")(x2)
outputs = layers.Dense(1, name="predictions")(x3)
MLmodel = keras.Model(inputs=inputs, outputs=outputs)

MLmodel.summary()

train_X = designMat_data
train_Y = Zmat[ind_data]
cv_X = designMat_cv
cv_Y = Zmat[ind_cv]


MLmodel.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    metrics=["MeanSquaredError","MeanAbsoluteError"],
)

print("fit!")
history = MLmodel.fit(train_X, train_Y, batch_size=1000, epochs=700, validation_split=0.2)
print("eval! (data+cv)")
fit_scores = MLmodel.evaluate(train_X, train_Y, verbose=2)
test_scores = MLmodel.evaluate(cv_X, cv_Y, verbose=2)
ML_prediction_Y_datapt = MLmodel.predict(train_X)
ML_prediction_Y_cv = MLmodel.predict(cv_X)

# ML: figures
fig3, ((ax01, ax02), (ax03, ax04)) = plt.subplots(2,2)
ax01.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=train_Y)
ax01.set_title("datapoint:Obs")
ax02.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=ML_prediction_Y_datapt)
ax02.set_title("datapoint:ModelFit")
ax03.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=cv_Y)
ax03.set_title("crossval:Obs")
ax04.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=ML_prediction_Y_cv)
ax04.set_title("crossval:Predict")
plt.show()

# print("glm loss - data fit:", glmMSE_data, " cv:", glmMSE_CV)
print("glm loss - data fit: 13.688539581526333  cv: 46.01889436984242")
