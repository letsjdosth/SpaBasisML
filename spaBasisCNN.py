#STD
import math

#scipy family
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

# #FOR ML
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

def test_data_generator(n_knot=100, n_data=1000, n_cv=500, seed_val=12345):
    # set the places
    randGenerator = np.random.default_rng(seed=seed_val)
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

    # thin plate splinte basis(below:TPS)
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
    
    # response matrix
    Zmat_data = Zmat[ind_data]
    Zmat_cv = Zmat[ind_cv]
    return (designMat_data, Zmat_data, gridLoc_data, designMat_cv, Zmat_cv, gridLoc_cv)
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
        green_mat = np.zeros((10,10,3))
        rgb_image = np.dstack((red_mat, green_mat, blue_mat))
        return_rgb_image_list.append(rgb_image)
    # print(return_rgb_image_list_data[0].shape) #(10, 10, 3)
    return return_rgb_image_list

if __name__ == "__main__":
    (designMat_data, Zmat_data, gridLoc_data, designMat_cv, Zmat_cv, gridLoc_cv) = test_data_generator()

    gray_image_list_data = to_gray_image(designMat_data, 2, (10,10))
    gray_image_list_cv = to_gray_image(designMat_cv, 2, (10,10))


    # gray image plot
    # fig1, ((ax101, ax102), (ax103, ax104)) = plt.subplots(2,2)
    # print(gray_image_list_data[0])
    # ax101.matshow(gray_image_list_data[0], cmap='gray')
    # ax102.matshow(gray_image_list_data[1], cmap='gray')
    # ax103.matshow(gray_image_list_data[2], cmap='gray')
    # ax104.matshow(gray_image_list_data[3], cmap='gray')
    # plt.show()



    rgb_image_list_data = from_gray_to_rgb_image_Thinplate(gray_image_list_data)
    rgb_image_list_cv = from_gray_to_rgb_image_Thinplate(gray_image_list_cv)

    #rgb image plot
    for i in range(0,30,10):
        fig2, ((ax201, ax202, ax203, ax204, ax205), (ax206, ax207, ax208, ax209, ax210)) = plt.subplots(2,5)
        ax201.imshow(rgb_image_list_data[i])
        ax202.imshow(rgb_image_list_data[i+1])
        ax203.imshow(rgb_image_list_data[i+2])
        ax204.imshow(rgb_image_list_data[i+3])
        ax205.imshow(rgb_image_list_data[i+4])
        ax206.imshow(rgb_image_list_data[i+5])
        ax207.imshow(rgb_image_list_data[i+6])
        ax208.imshow(rgb_image_list_data[i+7])
        ax209.imshow(rgb_image_list_data[i+8])
        ax210.imshow(rgb_image_list_data[i+9])
        plt.show()

