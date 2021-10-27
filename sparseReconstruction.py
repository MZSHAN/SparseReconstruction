import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from EpipolarHelpers import *
from utils import displayEpipolarF, epipolarMatchGUI


def load_corresponding_points(path):
    """
    Function to load the point correspondences from a npz file

    Args:
        path (string): path to the npz file

    Returns:
        Tuple of 2d numpy array with corresponding points
    """
    points = np.load(path)
    return points['pts1'], points['pts2']


def load_camera_intrinsics(path):
    intrinsics = np.load(path)
    return intrinsics['K1'], intrinsics['K2']


def plot_world_coordinates(world_coordinates):
    """
    Function to plot 3D world cordinates

    Args:
        world_coordinates - (N x 3) matrix with world coordinates
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    def init():
        ax.scatter(world_coordinates[:,1], -world_coordinates[:,2], -world_coordinates[:,0], alpha=0.6)
        return fig,


    def animate(i):
        ax.view_init(elev=0., azim=i)
        return fig,

    try:
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)

        writergif = animation.PillowWriter(fps=30)
        anim.save('reconstructed.gif',writer=writergif)
    except TypeError as e:
        pass

#    plt.show()


def get_epipolar_correspondences(x1, y1, image1, image2, fundamental_matrix):
    SP1, SP2 = [], []
    for i in range(x1.shape[0]):
        x2, y2 = epipolarCorrespondence(image1, image2, x1[i], y1[i], fundamental_matrix)
        SP1.append([x1[i], y1[i]])
        SP2.append([x2, y2])
    SP1 = np.array(SP1)
    SP2 = np.array(SP2)

    return SP1, SP2


if __name__ == "__main__":
    pts1, pts2 = load_corresponding_points("data/some_corresp.npz")
    cam_intrinsic1, cam_intrinsic2 = load_camera_intrinsics("data/intrinsics.npz")

    image1 = plt.imread("data/im1.png")
    image2 = plt.imread("data/im2.png")
    scaling_factor = max(image1.shape)

    #fundamental_matrix = ransac_fundamental_matrix(pts1, pts2, scaling_factor)

    fundamental_matrix = get_fundamental_matrix_eight_point(pts1, pts2, scaling_factor)

    #displayEpipolarF(image1, image2, fundamental_matrix)

     # Internally calls epipolar_correspondence(image1, image2, *pts1[0], fundamental_matrix)
     # epipolarMatchGUI(image1, image2, fundamental_matrix)

    essential_matrix = get_essential_matrix(fundamental_matrix, cam_intrinsic1, cam_intrinsic2)

    projection_mat1, projection_mat2 = get_projection_matrices(essential_matrix, cam_intrinsic1, cam_intrinsic2,
                                                               pts1[0], pts2[0])

    # Loads points in image 1 and find correspondences
    coords = np.load('data/templeCoords.npz')
    x1, y1 = coords['x1'][:, 0], coords['y1'][:, 0]
    SP1, SP2 = get_epipolar_correspondences(x1, y1, image1, image2, fundamental_matrix)

    world_coords = triangulate(projection_mat1, SP1, projection_mat2, SP2)

    plot_world_coordinates(world_coords)
