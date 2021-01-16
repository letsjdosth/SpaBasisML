import math

import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky


def test_data_generator(n_knot=100, n_data=1000, n_cv=500, seed_val=12345, dist=""):
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

    Zmat_intensity = 0
    Zmat = 0
    # trasform to each distribution
    if dist == "poisson":
        Zmat_intensity = np.exp(Wmat + np.matmul(Xmat_complete, MaternCov_params['beta']))
        Zmat = randGenerator.poisson(lam=Zmat_intensity)
        # print(Zmat.size) #1600
    elif dist == "gaussian":
        Zmat_intensity = Wmat + np.matmul(Xmat_complete, MaternCov_params['beta'])
        Zmat = 8*Zmat_intensity
    elif dist == "binomial":
        Zmat_intensity = 1 / (1 +np.exp(-(Wmat + np.matmul(Xmat_complete, MaternCov_params['beta'])))) 
        Zmat = randGenerator.binomial(1, Zmat_intensity)
    else:
        raise ValueError(str(dist)+" is not implemented yet.")

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

if __name__=="__main__":
    import matplotlib.pyplot as plt

    (designMat_data, Zmat_data, gridLoc_data, designMat_cv, Zmat_cv, gridLoc_cv) = test_data_generator(dist="binomial")
    
    # EDA: figures
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.scatter(gridLoc_data[:,0], gridLoc_data[:,1], c="blue", s=Zmat_data)
    ax1.set_title("Zmat_data")
    ax2.scatter(gridLoc_cv[:,0], gridLoc_cv[:,1], c="blue", s=Zmat_cv)
    ax2.set_title("Zmat_cv")
    plt.show()
