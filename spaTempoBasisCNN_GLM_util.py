import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky


# input : response_data[y1,...], lonlat_design_mat:[(covariates, long, lat, time),...]
# 
# def1: lonlat_design_mat:[...], space_knot_position:[(long,lat),...], time_knot_position[(time)] 
#   -> design matrix:[(response,covariates, spatial-basis1,...,time basis1,...),...]
# def2: (CNN fit with dropout) response, design matrix, layer structure -> last layer val
# def3: (CNN prediction with dropout) prediction_position(covariates, spatial-basis1,...,time basis1,...)->last layer val
# def4: (GLM fit using MCMC) last layer val, covariate -> summary values

def simulate_data(n_data, n_cv, true_coefficient, dist="poisson", seed_val=12345):

    # set the places
    randGenerator = np.random.default_rng(seed=seed_val)
    gridLoc_data = randGenerator.random((n_data,2))
    gridTime_data = randGenerator.random((n_data,1))
    gridLoc_cv = randGenerator.random((n_cv,2))
    gridTime_cv = randGenerator.random((n_cv,1))
    gridLoc_complete = np.concatenate((gridLoc_data, gridLoc_cv), axis=0)
    gridTime_complete = np.concatenate((gridTime_data, gridTime_cv), axis=0)

    # make covariates
    Xmat_data = randGenerator.random((n_data,2))
    Xmat_cv = randGenerator.random((n_cv,2))
    Xmat_complete = np.concatenate((Xmat_data, Xmat_cv), axis=0)
    # print(Xmat_complete.shape) #(1600,2)

    # distance matrix
    distMatLoc_complete = cdist(gridLoc_complete, gridLoc_complete, 'euclidean')
    distMatTime_complete = cdist(gridTime_complete, gridTime_complete, 'euclidean')
    
    #variance parameters
    param_overall_variance = 1
    param_temporal_corr = 500 #50, 500, 200
    param_spatial_corr = 2.5 #2.5, 25
    param_space_time_decay = 0.5 #0.75, 0.5, 0.95
    param_beta = true_coefficient

    Cov_mat_part1 = param_overall_variance / (param_temporal_corr * distMatTime_complete**2 + 1)**param_space_time_decay
    Cov_mat_part2 = np.exp(-param_spatial_corr * distMatLoc_complete / (param_temporal_corr * distMatTime_complete**2 + 1)**(param_space_time_decay /2))
    Cov_mat = Cov_mat_part1 * Cov_mat_part2
    print(Cov_mat.shape)


    # print(distMat_complete.shape) #(1600,1600)

    # Generate Data following model
    Cov_seed = randGenerator.normal(loc=0.0, scale=1.0, size=n_data + n_cv)
    Wmat = np.matmul(cholesky(Cov_mat).T, Cov_seed)
    # print(Wmat.shape) #(1600,)
    # print(np.max(Wmat), np.min(Wmat))

    Zmat_intensity = 0
    Zmat = 0
    # trasform to each distribution
    if dist == "poisson":
        Zmat_intensity = np.exp(Wmat + np.matmul(Xmat_complete, param_beta))
        Zmat = randGenerator.poisson(lam=Zmat_intensity)
        # print(Zmat.size) #1600
    elif dist == "gaussian":
        Zmat_intensity = Wmat + np.matmul(Xmat_complete, param_beta)
        Zmat = Zmat_intensity
    elif dist == "binomial":
        Zmat_intensity = 1 / (1 +np.exp(-(Wmat + np.matmul(Xmat_complete, param_beta)))) 
        Zmat = randGenerator.binomial(1, Zmat_intensity)
    else:
        raise ValueError(str(dist)+" is not implemented yet.")
    
    return Xmat_data, Zmat[0:n_data,], gridLoc_data, gridTime_data, Xmat_cv, Zmat[n_data:,], gridLoc_cv, gridTime_cv

def convert_latlontime_to_basis(covlonlattime_design_mat, lonlat_knot, time_knot):
    location_data = covlonlattime_design_mat[:, -3:-1]
    time_data = covlonlattime_design_mat[:, -1]

    location_dist = cdist(location_data, lonlat_knot, 'euclidean')
    # time_dist = cdist(time_data.reshape(len(time_data),1), time_knot.reshape(len(time_knot),1), 'euclidean')
    
    #loc: thin plate spline
    tps_location_basis = (location_dist**2) * (np.log(location_dist))

    #time: harmonic
    harmonic_time_basis_cos = np.cos(2 * np.pi * time_data.reshape(len(time_data),1) * time_knot.reshape(1,len(time_knot)))
    harmonic_time_basis_sin = np.sin(2 * np.pi * time_data.reshape(len(time_data),1) * time_knot.reshape(1,len(time_knot)))
    #r 코드에서는 cos sin cos sin 번갈아가며 끼우는데.. 귀찮다

    covbasis_design_mat = np.concatenate(
        (covlonlattime_design_mat[:,:-3],tps_location_basis,
        harmonic_time_basis_cos, harmonic_time_basis_sin), axis=1)
    
    return covbasis_design_mat




if __name__=="__main__":
    #example 1
    test1_covlonlattime_design_mat = np.array([
        #cov  lon  lat  time
        [1.3, 0.1, 0.5, 0.1],
        [0.8, 0.3, 0.8, 0.4],
        [0.6, 0.5, 0.5, 0.8]
    ])
    # print(test1_covlonlattime_design_mat[:,-3:-1])

    test1_lonlat_knot = np.array([[0.25,0.25],[0.75,0.75]])
    test1_time_knot = np.array([0.25, 0.75])
    test1_covbasis_design_mat= convert_latlontime_to_basis(test1_covlonlattime_design_mat, test1_lonlat_knot, test1_time_knot)
    print(test1_covbasis_design_mat.shape) #(3,7)


    #example 2
    test2_true_coeff = [2,2]
    test2_designMat_data, test2_Zmat_data, test2_gridLoc_data, test2_gridTime_data, test2_designMat_cv, test2_Zmat_cv, test2_gridLoc_cv, test2_gridTime_cv = simulate_data(100,50, test2_true_coeff)
    print(test2_designMat_data.shape, test2_Zmat_data.shape, test2_gridLoc_data.shape, test2_gridTime_data.shape,
        test2_designMat_cv.shape, test2_Zmat_cv.shape, test2_gridLoc_cv.shape, test2_gridTime_cv.shape)

    test2_covlonlattime_design_mat = np.c_[test2_designMat_data, test2_gridLoc_data, test2_gridTime_data]
    test2_conlonlattime_cv_mat = np.c_[test2_designMat_cv, test2_gridLoc_cv, test2_gridTime_cv]
    print(test2_covlonlattime_design_mat.shape, test2_conlonlattime_cv_mat.shape)#(100,5)(50,5) (covariate 2 + lon + lat + time)

    test2_lonlat_knot = np.array([[0.25,0.25],[0.75,0.75]])
    test2_time_knot = np.array([0.25, 0.75])
    test2_covbasis_design_mat = convert_latlontime_to_basis(test2_covlonlattime_design_mat, test2_lonlat_knot, test2_time_knot)
    test2_covbasis_cv_mat = convert_latlontime_to_basis(test2_conlonlattime_cv_mat, test2_lonlat_knot, test2_time_knot)
    print(test2_covbasis_design_mat.shape, test2_covbasis_cv_mat.shape)#100,8, 50,8 (covariate 2 + spatial basis 2 + time basis 2*2 (using harmonic basis))