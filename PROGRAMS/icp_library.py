import numpy as np
import numpy.linalg as la
import calibration_library as cal
import collections

def find_closest_point(point, vertices, triangles):
    """
    Find the closest point on the mesh surface defined by vertices and triangles to the given point.
    
    Parameters:
    point: Given one point as a NumPy array.
    vertices: Coordinates of vertices as a NumPy array.
    triangles: Indices of vertex coordinates for each triangle as a NumPy array.

    Returns:
    minpoint: the closest point to the given point on the surface mesh
    """
    num_triangles = triangles.shape[1]
    c_ij = np.zeros([3, num_triangles])
    S = np.zeros([3, 2])

    for i in range(num_triangles):
        # p, q, r = vertices[:, triangles[:, i]]
        p_index = int(triangles[0][i])
        q_index = int(triangles[1][i])
        r_index = int(triangles[2][i])

        p = vertices[:, p_index]
        q = vertices[:, q_index]
        r = vertices[:, r_index]
        # for a given triangle, print the 3D location of three vertices
        # print(p_index, q_index, r_index)
        # print(p,q,r)

        for j in range(3):
            S[j][0] = q[j] - p[j]
            S[j][1] = r[j] - p[j]
        b = point - p
        soln = la.lstsq(S, b, rcond=None)
        l = soln[0][0]
        m = soln[0][1]

        mid = p + l * (q - p) + m * (r - p)

        if l >= 0 and m >= 0 and l + m <= 1:
            c_star = mid
        elif l < 0:
            c_star = project_on_segment(mid , r, p)
        elif m < 0:
            c_star = project_on_segment(mid , p, q)
        else:  # l + m > 1
            c_star = project_on_segment(mid , q, r)

        c_ij[:, i] = c_star[:]

    distance = la.norm(point - c_ij[:, 0])
    minpoint = c_ij[:, 0]

    for i in range((c_ij).shape[1]):
        dist = la.norm(point - c_ij[:, i])

        if dist < distance:
            distance = dist
            minpoint = c_ij[: ,i]

    return minpoint

def project_on_segment(c, p, q):
    """
    Project point c onto the line segment defined by endpoints p and q.

    Parameters:
    c: The point to project as a NumPy array.
    p, q: Endpoints of the line segment as NumPy arrays.

    Returns:
    c_star: The projected point on the line segment.
    """
    # compute the scalar projection of c onto the line defined by p and q
    lambda_ = np.dot(c - p, q - p) / np.dot(q - p, q - p)

    # clamp lambda to lie within the segment
    lambda_seg = max(0, min(lambda_, 1))

    # compute the actual projection point on the segment
    c_star = p + lambda_seg * (q - p)

    return c_star

def calc_difference(c_k_points, d_k_points):
    """
    Calculates the Euclidean distance between corresponding points in two point clouds.

    Parameters:
    c_k_points: Closest point on surface to tip in each frame
    d_k_points: Position of tip in each frame

    Returns: 
    1D array with distances between each pair of corresponding points.
    """
    dist = np.zeros(np.shape(c_k_points)[0])
    for i in range(np.shape(c_k_points)[0]):
        dist[i] = np.linalg.norm(d_k_points[i] - c_k_points[i])

    return dist

def transform_tip_positions(tip_positions, frame_transformation):
    """
    Transforms tip positions with the given frame transformation. The func. will be useful in PA#4, In PA#3 we assume F_reg is I.
    param tip_positions: Array containing positions of tip with respect to rigid body B
    param frame_transformation: Frame transformation

    return: transformed_tip_positions: Transformed array of points

    """
    # for i in range(np.shape(tip_positions.data)[1]):

    #     tip_positions.data[:, i] = cal.setRegistration.apply_transformation_single_pt(tip_positions, frame_transformation)

    # return tip_positions
    transformed_tip_pos = []
    for i in range(len(tip_positions)):
        registration = cal.setRegistration()
        new = registration.apply_transformation_single_pt(tip_positions[i], frame_transformation)
        transformed_tip_pos.append(new)
    return transformed_tip_pos

def findClosestPoints(vertices, triangles, startPoints):
    """
    Finds the registration transformation between a rigid reference body B and the bone using an iterative closest point finding algorithm.

    :param vertices: Vertex coordinates of the mesh.
    :param triangles: Indices of vertices for each mesh triangle.
    :param startPoints: Positions of the rigid body's tip of rigid body A in reference B coordinates.

    :type vertices: np.array of np.float64, shape (3, N)
    :type triangles: np.array of np.float64, shape (3, M)
    :type startPoints: pc.PointCloud

    :return: Closest points and registration frame between the bone and the reference body B.
    """
    registrationFrame = np.identity(4) # initial transformation assumption
    maxIterations = 20
    previousError = collections.deque(maxlen=2) # deque of error
    previousError.append(0)
    registration = cal.setRegistration()


    for iteration in range(maxIterations):
        transformedPoints = transform_tip_positions(startPoints, registrationFrame)
        #transformedPoints = cal.setRegistration.apply_transformation_single_pt(startPoints, registrationFrame)
        allClosestPoints = []

        for point in transformedPoints:
            closestPoint = find_closest_point(point, vertices, triangles)
            allClosestPoints.append(closestPoint)

        delta_Frame = registration.calculate_3d_transformation(transformedPoints, np.array(allClosestPoints))
        newFrame = np.matmul(delta_Frame, registrationFrame)

        if hasConverged(1e-4, registrationFrame, newFrame, previousError):
            return np.array(allClosestPoints), newFrame

        registrationFrame = newFrame

    return np.array(allClosestPoints), registrationFrame


def hasConverged(tolerance, oldFrame, newFrame, errorHistory):
    """
    Determines if the frame transformation is within a specified tolerance.

    :param tolerance: Tolerance threshold for sum of squared differences.
    :param oldFrame: Previous frame transformation.
    :param newFrame: Current frame transformation.
    :param errorHistory: History of previous errors to assess convergence.

    :type tolerance: float
    :type oldFrame: 4x4 matrix
    :type newFrame: 4x4 matrix
    :type errorHistory: collections.deque

    :return: Whether the difference is within the tolerance or not.
    :rtype: bool
    """
    error = sum((oldFrame[:3, 3][i] - newFrame[:3, 3][i]) ** 2 +
                sum((oldFrame[:3, :3][i][j] - newFrame[:3, :3][i][j]) ** 2 for j in range(3))
                for i in range(3))

    if error < tolerance or (errorHistory and abs(error - errorHistory[0]) < tolerance):
        return True

    errorHistory.append(error)
    return False

