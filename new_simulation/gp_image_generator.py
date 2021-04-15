import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

def make_grid(grid_shape_tuple):
    ''' 
    generate grid on [0,grid_shape_tuple[0]]X[0,grid_shape_tuple[1]] in Z^2
    choose mark points as left-lower vertex (in picture, left-upper vertex)
    '''
    gridLoc = []
    for i in range(grid_shape_tuple[0]):
        for j in range(grid_shape_tuple[1]):
            # gridLoc.append([i/grid_shape_tuple[0],j/grid_shape_tuple[1]]) #<- normalize version
            gridLoc.append([i,j])
    return gridLoc


def gp_image_generator(num_image, grid_shape_tuple, matern_param_dict, seed_val=20210415):
    '''
    generate image on [0,grid_shape_tuple[0]]X[0,grid_shape_tuple[1]] in Z^2
    the value is normally distributed, structured by matern covariance form
    input:
        num_image : int
        grid_shape_tuple : tuple(int,int). For example, to make a 10 by 10 image, set this argument to (10,10)
        matern_covariance_dict: dictionary{"phi": float, "sigma2": float}. The keys should not be changed (up to order.)
            phi: strengthness of correlation among near points (higher phi -> more stronger corr)
            sigma2: general variance
        seed_val : int ir float
    
    output:
        numpy array[image_0, image_1, ..., image_'num_image']
        where image_i : 'grid_shape_tuple'-shaped matrix
    '''

    # set the places
    randGenerator = np.random.default_rng(seed=seed_val)
    gridLoc = make_grid(grid_shape_tuple)

    # Matern covariance structure seting
    if "phi" not in matern_param_dict.keys():
        print('hi')
        raise KeyError("matern_param_dict should include the parameter key:value pair -> 'phi':x")
    if "sigma2" not in matern_param_dict.keys():
        raise KeyError("matern_param_dict should include the parameter key:value pair -> 'sigma2':x")

    distMatLoc = cdist(gridLoc, gridLoc, 'euclidean')
    MaternCov_mat = (1 + (5**0.5 * distMatLoc / matern_param_dict["phi"]) +(5 * distMatLoc**2) / (3*matern_param_dict["phi"]**2)) * \
        np.exp(-(5**0.5 *(distMatLoc/matern_param_dict["phi"])))
    MaternCov_mat *= matern_param_dict["sigma2"]


    #generate images
    image_list = []
    for _ in range(num_image):
        MaternCov_TSseed = randGenerator.normal(loc=0.0, scale=1.0, size=MaternCov_mat.shape[0])
        gp_image = np.matmul(cholesky(MaternCov_mat).T, MaternCov_TSseed)
        gp_image_vector_form = np.array(gp_image)
        gp_image_matrix_form = np.reshape(gp_image_vector_form, grid_shape_tuple)
        image_list.append(gp_image_matrix_form) #gaussian
        # image_list.append(np.exp(gp_image_matrix_form)) #poisson

    return np.array(image_list)

def image_filter_generator(loc_knots, grid_shape_tuple, basis="invQuad", **kwargs):
    '''
    generate image filter on [0,grid_shape_tuple[0]]X[0,grid_shape_tuple[1]] in Z^2
    input:
        loc_knots: list of 2-dim-list. the centers of the kernels.
        grid_shape_tuple: tuple(int,int). recommendation: for additional works, set this argument as same as one of image generater
        basis: choose from below
            currently implemented basis : 
                "invQuad"
                    needs additional argument 'phi: float', controlling the decay of the kernel as distance grows.
                    if phi is larger, the value of kernel becomes more fastly decrease as distance grows.
                        example : image_filter_generator(..., basis="invQuad", phi=0.05)
        
    output:
        numpy array[filter_using 'loc_knots_0', filter_using 'loc_knots_1',...]
        where filter_i : 'grid_shape_tuple'-shaped matrix
    '''

    gridLoc = np.array(make_grid(grid_shape_tuple))
    distMat_knot_grid = np.array(cdist(loc_knots, gridLoc, 'euclidean'))
    #distmat[k][p]: from k-th knots to p-th gridpoint
    basis_mat = 0
    
    # TPSbasis
    # TPSmat = (distMat_knot_grid**2) * (np.log(distMat_knot_grid)) #issue: 1. 제곱이 있나? 2. knot에 거리 0인 점 있으면 error
   
    #invQuadBasis
    if basis == "invQuad":
        if "phi" in kwargs.keys():
            phi = kwargs["phi"]
        else:
            raise ValueError("please put additional argument 'phi' to control the decay of this kernel")
        basis_mat = 1/ (1+(phi*distMat_knot_grid)**2)
    
    else:
        raise ValueError("the basis is not yet implemented:", basis)
    basis_mat = np.reshape(basis_mat, (len(loc_knots),grid_shape_tuple[0],grid_shape_tuple[1]))
    return basis_mat

def image_transformer(image_array, filter_array):
    ''' construct f(image) with basis
        now:
            f(image) = sum(image*filter) (*: elementwise multiplication)
        
        input: 
            image_array (from gp_image_generator())
            filter_array (from image_filter_generator())
        output: 
            (number of images) by (number_of_filters) matrix
    '''
    feature_list = []
    for image in image_array:
        for filt in filter_array:
            filtered_image = image*filt
            feature_list.append(np.sum(filtered_image))
    feature_array = np.array(feature_list)
    feature_array = np.reshape(feature_array, (len(image_array), len(filter_array)))
    return feature_array


if __name__=="__main__":
    image_shape = (40,40)
    image_array = gp_image_generator(5, grid_shape_tuple=image_shape, matern_param_dict={"phi" : 15, "sigma2" : 1})
    # fig1, ((ax101, ax102), (ax103, ax104)) = plt.subplots(2,2)
    # ax101.matshow(image_array[0], cmap='gray') #흰색이 high value
    # ax102.matshow(image_array[1], cmap='gray')
    # ax103.matshow(image_array[2], cmap='gray')
    # ax104.matshow(image_array[3], cmap='gray')
    # plt.show()
    
    basis_array = image_filter_generator([[0, 0], [40, 40], [10, 30], [20, 20]], grid_shape_tuple=image_shape, basis="invQuad", phi=0.1)
    # fig1, ((ax101, ax102), (ax103, ax104)) = plt.subplots(2,2)
    # ax101.matshow(basis_array[0], cmap='gray') #흰색이 high value
    # ax102.matshow(basis_array[1], cmap='gray')
    # ax103.matshow(basis_array[2], cmap='gray')
    # ax104.matshow(basis_array[3], cmap='gray')
    # plt.show()

    feature = image_transformer(image_array, basis_array)

    #for test
    choose_image_index = 1
    print(feature[choose_image_index])
    fig1, ((ax101, ax102), (ax103, ax104), (ax105, ax106), (ax107, ax108)) = plt.subplots(4,2)
    ax101.matshow(image_array[choose_image_index], cmap='gray') #흰색이 high value
    ax102.matshow(basis_array[0], cmap='gray')
    ax103.matshow(image_array[choose_image_index], cmap='gray')
    ax104.matshow(basis_array[1], cmap='gray')
    ax105.matshow(image_array[choose_image_index], cmap='gray')
    ax106.matshow(basis_array[2], cmap='gray')
    ax107.matshow(image_array[choose_image_index], cmap='gray')
    ax108.matshow(basis_array[3], cmap='gray')
    plt.show()
