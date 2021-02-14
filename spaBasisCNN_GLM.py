from os import getpid
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from glm_MCMC import glmMCMC


from test_data_generator import test_data_generator

# input data
(designMat_data, Zmat_data, gridLoc_data, designMat_cv, Zmat_cv, gridLoc_cv) = test_data_generator(dist="poisson")
X_train_covariate = np.reshape(designMat_data[:,:2], (1000,2,1))
X_train_basis = np.reshape(designMat_data[:,2:], (1000,100,1))
X_cv_covariate = np.reshape(designMat_cv[:,:2], (500,2,1))
X_cv_basis = np.reshape(designMat_cv[:,2:], (500,100,1))

def procedure_eachcore(result_queue, CNN_model_prediction):
    proc_pid = getpid()
    print("pid: ", proc_pid, "start!")
    
    #using glmMCMC by hand
    glm_designmat = np.c_[np.ones(1000), np.reshape(X_train_covariate,(1000,2)), CNN_model_prediction]

    glm_inst = glmMCMC(glm_designmat, Zmat_data, 'poisson', np.zeros(glm_designmat.shape[1]))
    # args: design_mat_data, response_data, glmclass, initial_param, rng_seed=2021)
    glm_inst.generate_samples_with_dimgroup(100000, group_thres=3, pid=proc_pid)
    glm_inst.burnin(20000)
    glm_inst.thinning(4)
    result_queue.put(glm_inst)
    print(glm_inst.get_sample_mean()) #true: covariate 2 terms: [2,2]
    glm_inst.show_traceplot()
    glm_inst.show_hist()

if __name__=="__main__":
    #~CNN level
    #CNN model
    image_input = layers.Input(shape=(100,1))
    conv1 = layers.Conv1D(16, 6, activation='relu')(image_input)
    conv1 = layers.MaxPooling1D()(conv1)
    conv1 = layers.Dropout(0.2)(conv1)
    conv2 = layers.Conv1D(32, 6, activation='relu')(conv1)
    conv2 = layers.MaxPooling1D()(conv2)
    conv2 = layers.Dropout(0.2)(conv2)
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


    #~glm, multiprocessing
    #detailed tuning should be conducted at glm_MCMC.py

    core_num = 2
    process_vec = []
    proc_queue = mp.Queue()

    CNN_model_last_layer_predictions = []
    for i in range(core_num):
        CNN_model_prediction = CNN_model.predict(X_train_basis)
        CNN_model_last_layer_predictions.append(np.array(CNN_model_prediction[0]))
        print("prediction:", i)
    
    for i in range(core_num):
        process_unit = mp.Process(target=procedure_eachcore,
        args=(proc_queue, CNN_model_last_layer_predictions[i]))

        process_vec.append(process_unit)

    for unit_proc in process_vec:
        unit_proc.start()

    mp_result_vec = []
    for _ in range(core_num):
        each_result = proc_queue.get()
        mp_result_vec.append(each_result)
    
    for unit_proc in process_vec:
        unit_proc.join()

    print("exit multiprocessing")
    





    # #using pymc3
        # with pm.Model() as GLM_model:
    #     # define priors, weakly informative Normal
    #     b0 = pm.Normal("b0_intercept", mu=0, sigma=100)
    #     b1 = pm.Normal("b1_covariate1", mu=0, sigma=100)
    #     b2 = pm.Normal("b2_covariate2", mu=0, sigma=100)
    #     b3 = pm.Normal("b3_basis", mu=0, sigma=100)
    #     b4 = pm.Normal("b4_basis", mu=0, sigma=100)
    #     b5 = pm.Normal("b5_basis", mu=0, sigma=100)
    #     b6 = pm.Normal("b6_basis", mu=0, sigma=100)
    #     b7 = pm.Normal("b7_basis", mu=0, sigma=100)
    #     b8 = pm.Normal("b8_basis", mu=0, sigma=100)
    #     b9 = pm.Normal("b9_basis", mu=0, sigma=100)
    #     b10 = pm.Normal("b10_basis", mu=0, sigma=100)
    #     b11 = pm.Normal("b11_basis", mu=0, sigma=100)
    #     b12 = pm.Normal("b12_basis", mu=0, sigma=100)
    #     b13 = pm.Normal("b13_basis", mu=0, sigma=100)
    #     b14 = pm.Normal("b14_basis", mu=0, sigma=100)

    #     # define linear model and exp link function
    #     theta = (
    #         b0
    #         + b1 * X_train_covariate[:,0]
    #         + b2 * X_train_covariate[:,1]
    #         + b3 * CNN_model_last_layer_prediction[:,0]
    #         + b4 * CNN_model_last_layer_prediction[:,1]
    #         + b5 * CNN_model_last_layer_prediction[:,2]
    #         + b6 * CNN_model_last_layer_prediction[:,3]
    #         + b7 * CNN_model_last_layer_prediction[:,4]
    #         + b8 * CNN_model_last_layer_prediction[:,5]
    #         + b9 * CNN_model_last_layer_prediction[:,6]
    #         + b10 * CNN_model_last_layer_prediction[:,7]
    #         + b11 * CNN_model_last_layer_prediction[:,8]
    #         + b12 * CNN_model_last_layer_prediction[:,9]
    #         + b13 * CNN_model_last_layer_prediction[:,10]
    #         + b14 * CNN_model_last_layer_prediction[:,11]
    #     )

    #     ## Define Poisson likelihood
    #     y = pm.Poisson("y", mu=np.exp(theta), observed=Zmat_data)
    
    # with GLM_model:
    #     glm_model_fit = pm.sample(1000, tune=1000, return_inferencedata=True, cores=1) #<-do not operate on windows10 (multiprocessing problem)

    # az.plot_trace(glm_model_fit)

