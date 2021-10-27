import math
import numpy as np
import matplotlib.pyplot as plt
import utils


# Function has been tested
def triangulate(projection_matrix1, pts1, projection_matrix2, pts2):
    """
    Function takes a the projection matrices of two cameras and the corresponding points in the pixel space of the
    2 camera. It returns the 3D location of the corresponding points.
    Reprojection of Point P in the pixel space using the projection matrix. This should be equal to the actual pixel
    location in Image. Solve:
        - Linear constraints from p X MP = 0 ; p is pixel coordinate and P is 3d point
        - Minimizing non linear reprojection error

    Args:
        projection_matrix1: (3 X 4) projection matrix of camera 1
        pts1: (N x 2) matrix with 2D points in camera 1
        projection_matrix2: (3 X 4) projection matrix of camera 2
        pts2: (N x 2) matrix with corresponding 2D points in camera 2
    Returns:
        (N x 3) world coordinates corresponding to pts1 and pts2
    """
    # Get coordinates as column vectors - eases matrix operations
    x1, y1 = np.expand_dims(pts1[:, 0], axis = 1), np.expand_dims(pts1[:, 1], axis = 1)
    x2, y2 = np.expand_dims(pts2[:, 0], axis = 1), np.expand_dims(pts2[:, 1], axis = 1)


    # Calculate contraints for all points
    constraint1 = x1 * projection_matrix1[2] - projection_matrix1[0]
    constraint2 = y1 * projection_matrix1[2] - projection_matrix1[1]
    constraint3 = x2 * projection_matrix2[2] - projection_matrix2[0]
    constraint4 = y2 * projection_matrix2[2] - projection_matrix2[1]

    N = len(pts2)
    world_coords = np.zeros((N, 3))

    # triangulate
    for i in range(N):
        A = np.vstack((constraint1[i], constraint2[i], constraint3[i], constraint4[i]))
        U, S, Vh = np.linalg.svd(A)
        assert(Vh[-1].shape == (4, )) # TODO: Remove later
        P = Vh[-1]
        world_coords[i] = P[:3] / P[-1] # Convert from homogenous to world - 4D to 3D coordinates

    world_coords_homo = np.concatenate((world_coords, np.ones((N, 1))), axis=1) # Convert to homog for projection

    # compute reprojection error
    projected_pts1 = np.dot(projection_matrix1, world_coords_homo.T)
    projected_pts1 = np.transpose(projected_pts1[:2] / projected_pts1[2])
    projected_pts2 = np.dot(projection_matrix2, world_coords_homo.T)
    projected_pts2 = np.transpose(projected_pts2[:2] / projected_pts2[2])

    assert(pts1.shape == projected_pts1.shape)
    assert(pts2.shape == projected_pts2.shape)

    reprojection_error = np.sum((projected_pts1 - pts1)**2 + (projected_pts2 - pts2)**2)
    avg_reprojection_error = math.sqrt(reprojection_error) / len(pts1)
    print (f"The total reprojection error is {avg_reprojection_error}\n")

    return world_coords

# Function hs been tested
def camera2_extrisics_from_essential_matrix(essential_matrix):
    """
    Function take essential matrix as input and returns the 4 possible rotations and translations between the stereo
    cameras

    First the sigular value contraint of Essential Matrix is enforced
        Essential Matrix = [Tx]R. [Tx] is a skew symmetric matrix. 3D skew symmetric matrices have 2 equal singular values
        Since R is just a rotation, essential matrix is has 2 singular values which are equal

    E = [Tx]R = U(Sigma)V, then with W as skew symmetric matrix as shown below can be used to get t and R as shown below
    so that they satisfy [Tx]R = E
    But t and -t, R and R.T so obtained satisfy the properties. All 4 combinations are returned from this function, but
    only one is correct. Correct can be found by a pair of corresponding points downstream.

    Refer: https://en.wikipedia.org/wiki/Essential_matrix#Extracting_rotation_and_translation
            and Hartley Zisserman 9.6.2

    Args:
        essential_matrix: 3 X 3 essential matrix for camera pair
    Returs:
        (3 X 4 X 4) matrix with possible Extrinsics - rotation and translation between 2 cameras
    """
    # correct for equal singular values
    U, S, Vh = np.linalg.svd(essential_matrix)
    mean_s = S[0:2].mean()
    S = np.eye(len(S)) * mean_s; S[-1, -1] = 0
    essential_matrix = np.dot(U, np.dot(S, Vh))
    U, S, Vh = np.linalg.svd(essential_matrix) # Recalculate SVD


    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # This is one of the rotation matrices(W/W.T) - ensures that we have a projection and space is not flipped
    if np.linalg.det(np.dot(U, np.dot(W, Vh))) < 0:
        W = -W

    # All possible Extrinsic matrices are [UWV.T | u3], [UWV.T | -u3], [U(W.T)V.T | u3], [U(W.T)V.T | -u3]
    T = U[:, 2].reshape(-1, 1)/abs(U[:, 2]).max()
    R1 = np.dot(U, np.dot(W, Vh))
    R2 = np.dot(U, np.dot(W.T, Vh))

    Extrinsics = np.zeros((4, 3, 4))
    Extrinsics[0] = np.concatenate((R1, T), axis = 1)
    Extrinsics[1] = np.concatenate((R1, -T), axis = 1)
    Extrinsics[2] = np.concatenate((R2, T), axis = 1)
    Extrinsics[3] = np.concatenate((R2, -T), axis = 1)

    return Extrinsics


# Function hs been tested
def get_projection_matrices(essential_matrix, cam_intrinsic1, cam_intrinsic2, pt1, pt2):
    """
    Function returns the projection matrices of the two cameras in a stereo system
    Function first calculates 4 possible Rotations and Translations matrices from the essential matrix
    Out of the 4 possiblities, one is selected based on min projection error
    Both Camera intrincs along with rotation and translation give the Projection matrices for both cameras

    Args:
        essential_matrix: (3X3) essential matrix
        cam_intrinsic1: (3 x 3) matrix, intrinsic parameters for camera 1
        cam_intrinsic2: (3 x 3) matrix, intrinsic parameters for camera 2
        pt1: (2, ) numpy array, a 2D piont in image1
        pt2: (2, ) numpy array, a 2D piont in image2 corresponding to pt1

    returns:
        projection_mat1
    """
    poss_extrinsics = camera2_extrisics_from_essential_matrix(essential_matrix)

    # Make homogenous 2D i.e. 3D points
    pt1, pt2 = np.append(pt1, [1]), np.append(pt1, [1])
    pt1, pt2 = np.expand_dims(pt1, axis=1), np.expand_dims(pt2, axis=1)
    assert(pt1.shape == (3, 1))
    pt1, pt2 = np.linalg.inv(cam_intrinsic1).dot(pt1), np.linalg.inv(cam_intrinsic2).dot(pt2) # From pixel to image space
    pt1 = np.vstack((pt1, [[1]])) # Make 3D homogenous i.e. 4D points
    assert(pt1.shape == (4, 1))

    min_err, extrinsic2 = float('inf'), None
    for ext in poss_extrinsics:
        pred_pt2 = np.dot(ext, pt1)
        assert(pred_pt2.shape == pt2.shape)
        err = np.sum((pred_pt2 - pt2)**2)
        if err < min_err:
            min_err = err
            extrinsic2 = ext

    extrinsic1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis = 1)
    projection_mat1 = np.dot(cam_intrinsic1, extrinsic1)
    projection_mat2 = np.dot(cam_intrinsic2, extrinsic2)

    assert(projection_mat1.shape == (3, 4))
    assert(projection_mat2.shape == (3, 4))

    return projection_mat1, projection_mat2


# test using some corresp noisy.npz
def ransac_fundamental_matrix(pts1, pts2, normalization_factor):
    """
    Function calculates the best fundamental matrix through Ransac

    Ransac requires the function to be calculated using min num of points. Hence seven point algorithm is used.

    Args:
         pts1: (N x 2) Numpy array of 2D points from image 1
         pts2: (N x 2) Numpy array of corrsponding feature points from image2
         normalization_factor: maximum of width and height of the images
    Returns:
         ( 3 X 3) best estimate of the fundamental matrix
         indices of in inlier points from pts1, pts2
    """
    assert(pts1.shape == pts2.shape)
    N = len(pts1)
    threshold = 1

    # Cconvert points to homogenous coordinates
    homo_pts1 = np.hstack((pts1, np.ones((N, 1))))
    homo_pts2 = np.hstack((pts2, np.ones((N, 1))))
    fundamental_matrix, inliers, max_inliers = None, None, -1

    # Run ransac for 1000 iterations
    for i in range(1000):
        # calculate sample fundamental matrix using 7 points
        ind = np.random.randint(0, N, 7)
        sample_fundamental_matrices = sevenpoint(pts1[ind], pts2[ind], normalization_factor)

        for sample_fund_matrix in sample_fundamental_matrices:
            # Use sample fundamental matrix to get epipolar lines for all points
            epipolar_lines = np.dot(sample_fund_matrix, homo_pts1.T)
            assert(epipolar_lines.shape == (3, N))
            epipolar_lines = epipolar_lines / (epipolar_lines[0][:]**2 + epipolar_lines[1][:]**2)
            epipolar_lines = epipolar_lines.T

            # Calculate distance of points from epipolar lines
            distance = abs(np.sum(homo_pts2 * epipolar_lines, axis=1))
            curr_inliers = np.arange(N).reshape(N, 1)[distance < threshold]

            if len(curr_inliers) > max_inliers:
                inliers = curr_inliers
                max_inliers = len(curr_inliers)
                fundamental_matrix = sample_fund_matrix

    print ("For ransac fundamental matrix: Average #inlier is :", max_inliers / N)

    return fundamental_matrix, inliers

#This function has been taken from the internet.
#Returns list of estimated fundamental matrices
def sevenpoint(pts1, pts2, M):
    # normalize the coordinates
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    x1, y1, x2, y2 = x1 / M, y1 / M, x2 / M, y2 / M
    # normalization matrix
    T = np.array([[1. / M, 0, 0], [0, 1. / M, 0], [0, 0, 1]])

    A = np.transpose(np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(x1.shape))))

    # get F by SVD decomposition
    u, s, vh = np.linalg.svd(A)
    f1 = vh[-1, :]
    f2 = vh[-2, :]
    F1 = np.reshape(f1, (3, 3))
    F2 = np.reshape(f2, (3, 3))

    fun = lambda alpha: np.linalg.det(alpha * F1 + (1 - alpha) * F2)
    # get the coefficients of the polynomial
    a0 = fun(0)
    a1 = 2*(fun(1)-fun(-1))/3 - (fun(2)-fun(-2))/12
    a2 = (fun(1)+fun(-1))/2 - a0
    a3 = (fun(1)-fun(-1))/2 - a1
    # solve for alpha
    alpha = np.roots([a3, a2, a1, a0])

    Farray = [a*F1+(1-a)*F2 for a in alpha]
    # refine F
    Farray = [utils.refineF(F, pts1/M, pts2/M) for F in Farray]
    # denormalize F
    Farray = [np.dot(np.transpose(T), np.dot(F, T)) for F in Farray]

    return Farray


def _get_normalization_matrix(pts1, pts2, M):
    return np.diag([1.0/M, 1.0/M, 1.0])


def get_fundamental_matrix_eight_point(pts1, pts2, M):
    """
    Do not understand the geometric local refining part yet

    refine the solution by using local minimization.
    make a good solution better by locally minimizing a geometric cost function.
    call from eightpoint before unscaling F.
    """
    assert(len(pts1) == len(pts2), "Correspondence mismatch")

    norm_matrix = _get_normalization_matrix(pts1, pts2, M)
    #Not coverting pts to homogenous co-ordinates to save memory. Rather multiply by 2X2 norm matrix for same points
    norm_pt1 = np.dot(norm_matrix[0:2, 0:2], pts1.T).T
    norm_pt2 = np.dot(norm_matrix[0:2, 0:2], pts2.T).T

    x1, y1 = norm_pt1[:, 0], norm_pt1[:, 1]
    x2, y2 = norm_pt2[:, 0], norm_pt2[:, 1]
    feature_matrix = np.transpose(np.stack((x1*x2, y2*x1, x1, x2*y1, y1*y2, y1, x2, y2, np.ones(x1.shape))))

    u, s, vh = np.linalg.svd(feature_matrix)
    unconstrained_fund_matrix = vh[-1].reshape(3, 3)

    unconstrained_fund_matrix = utils.refineF(unconstrained_fund_matrix, norm_pt1, norm_pt2)

    U, S, Vh = np.linalg.svd(unconstrained_fund_matrix)
    S[-1] = 0
    sing_values = np.diag(S)
    unnormazized_fund_matrix = np.dot(U, np.dot(sing_values, Vh))

    fundamental_matrix = np.dot(norm_matrix.T, np.dot(unnormazized_fund_matrix, norm_matrix))
    return fundamental_matrix


def _get_epipolar_line_pixels(epipolar_line, im2):
    if epipolar_line[0] == 0 and epipolar_line[1] == 0:
        raise Exception("Line vector can not be zero")

    im_y, im_x, _ = im2.shape

    epipolar_coordinates = []
    if epipolar_line[0] != 0:
        for y in range(im_y):
            x = -(epipolar_line[1] * y + epipolar_line[2])/epipolar_line[0]
            epipolar_coordinates.append([round(x), y])
    else:
        for x in range(im_x):
            y = -(epipolar_line[0] * x + epipolar_line[2])/epipolar_line[1]
            epipolar_coordinates.append([x, round(y)])
    return np.array(epipolar_coordinates)


def test_epipolar_line(im1, im2, fundamental_matrix):
    x1, y1 = 475,  96

    print (f"Testing ..... plotting the epipolar line for {x1} and {y1} ")
    point = np.array([[x1, y1, 1]]).T

    epipolar_line = np.dot(fundamental_matrix.T, point)
    epipolar_line = np.squeeze(epipolar_line.T)

    epipolar_coordinates = _get_epipolar_line_pixels(epipolar_line, im2)

    plt.imshow(im2)
    plt.scatter(epipolar_coordinates[:,0], epipolar_coordinates[:, 1])
    plt.show()


def _valid_epipolar_coordinates(im2, epipolar_coordinates, center):
    ht, wd, _ = im2.shape
    valid_x = np.logical_and(epipolar_coordinates[:, 0] < wd - center, epipolar_coordinates[:, 0] >= center)
    valid_y = np.logical_and(epipolar_coordinates[:, 1] < ht - center, epipolar_coordinates[:, 1] >= center)
    valid_epipolar = epipolar_coordinates[np.logical_and(valid_x,  valid_y)]
    return valid_epipolar


def epipolar_correspondence(im1, im2, x1, y1, fundamental_matrix):
    """
    This is a noob version that uses L2 distances to find the feature poitns. Fails for similar corners in diff locations
    >>Since there is not much change in the 2 images, the points should be searched locally for best results here

    Ideally SIFT discriptors should be used
    """
    window_size = 11
    center = window_size//2
    sigma = 5

    point = np.array([[x1, y1, 1]]).T

    epipolar_line = np.dot(fundamental_matrix, point)
    epipolar_line = np.squeeze(epipolar_line.T)

    epipolar_coordinates = _get_epipolar_line_pixels(epipolar_line, im2)
    valid_coords = _valid_epipolar_coordinates(im2, epipolar_coordinates, center)

    target_patch = im1[y1-center:y1+center+1, x1-center:x1+center+1]

    #Generate a gaussian mask to weight error
    mask = np.ones((window_size, window_size))*center
    mask = np.repeat(np.array([range(window_size)]), window_size, axis=0) - mask
    mask = mask**2+np.transpose(mask)**2
    weight = np.exp(-0.5*mask/(sigma**2))
    weight /= np.sum(weight)

    correspond, min_error = None, float("inf")
    for x, y in valid_coords:
        source_patch = im2[y-center:y+center+1, x-center:x+center+1]

        error = ((target_patch - source_patch)**2).transpose(2, 0, 1)
        weighed_error = np.sum(np.multiply(error,  weight))

        if weighed_error < min_error:
            correspond = [x, y]
            min_error = weighed_error

    return correspond


def epipolarCorrespondence(im1, im2, x1, y1, F):
    # set the size of the window
    x1, y1 = int(round(x1)), int(round(y1))
    window_size = 11
    center = window_size//2
    sigma = 5
    search_range = 40

    # create gaussian weight matrix
    mask = np.ones((window_size, window_size))*center
    mask = np.repeat(np.array([range(window_size)]), window_size, axis=0) - mask
    mask = np.sqrt(mask**2+np.transpose(mask)**2)
    weight = np.exp(-0.5*(mask**2)/(sigma**2))
    weight /= np.sum(weight)

    if len(im1.shape) > 2:
        weight = np.repeat(np.expand_dims(weight, axis=2), im1.shape[-1], axis=2)

    # get the epipolar line
    p = np.array([[x1], [y1], [1]])
    l2 = np.dot(F, p)

    # get the patch around the pixel in image1
    patch1 = im1[y1-center:y1+center+1, x1-center:x1+center+1]
    # get the points on the epipolar line
    h, w, _ = im2.shape
    Y = np.array(range(y1-search_range, y1+search_range))
    X = np.round(-(l2[1]*Y+l2[2])/l2[0]).astype(np.int)
    valid = (X >= center) & (X < w - center) & (Y >= center) & (Y < h - center)
    X, Y = X[valid], Y[valid]

    min_dist = None
    x2, y2 = None, None
    for i in range(len(X)):
        # get the patch around the pixel in image2
        patch2 = im2[Y[i]-center:Y[i]+center+1, X[i]-center:X[i]+center+1]
        # calculate the distance
        dist = np.sum((patch1-patch2)**2*weight)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            x2, y2 = X[i], Y[i]

    return x2, y2


# Function has been tested
def get_essential_matrix(fundamental_matrix, cam_intrinsic1, cam_intrinsic2):
    essential_matrix = np.dot(cam_intrinsic2.T, np.dot(fundamental_matrix, cam_intrinsic1))
    return essential_matrix