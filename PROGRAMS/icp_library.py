import numpy as np
import numpy.linalg as la
import calibration_library as cal
import collections
from scipy.spatial import KDTree, cKDTree

def linear_search_closest_point(point, vertices, triangles):
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
    c_ij = np.zeros([3, num_triangles]) # closest point initialize with zeros 
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

def findClosestPoints(vertices, triangles, startPoints, searchMode):
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
    iteration = 0 
    maxIterations = 4
    previousError = collections.deque(maxlen=2) # deque of error
    previousError.append(0)
    registration = cal.setRegistration()
    bounding_boxes = construct_bounding_boxes(vertices, triangles)
    # root = center of all spheres, leaf = actual sphere  
    box_centers = [(box.min_corner + box.max_corner) / 2 for box in bounding_boxes]
    kdtree = KDTree(box_centers)

    # kdtree
    vertex_list = [vertices[:, i] for i in range(vertices.shape[1])]
    kdtree = KDTree(vertex_list)

    while iteration < maxIterations:
        transformedPoints = transform_tip_positions(startPoints, registrationFrame)
        allClosestPoints = []

        for point in transformedPoints:
            closestPoint = linear_search_closest_point(point, vertices, triangles)
            # Assuming bounding_boxes is a list of BoundingBox objects
            #closestPoint = find_closest_point_bbox(point, kdtree, bounding_boxes, vertices)
            allClosestPoints.append(closestPoint)

        delta_Frame = registration.calculate_3d_transformation(transformedPoints, np.array(allClosestPoints))
        newFrame = np.matmul(delta_Frame, registrationFrame)

        if hasConverged(1e-4, registrationFrame, newFrame, previousError):
            return np.array(allClosestPoints), newFrame

        registrationFrame = newFrame
        iteration += 1
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


class BoundingBox:
    def __init__(self, min_corner, max_corner, triangle_vertex_indices):
        self.min_corner = np.array(min_corner)
        self.max_corner = np.array(max_corner)
        self.triangle = triangle_vertex_indices  # The triangle or object reference

    def contains_point(self, point):
        return np.all(point >= self.min_corner) and np.all(point <= self.max_corner)
    
# def construct_bounding_boxes(vertices, triangles):
#     bounding_boxes = []
#     flat_triangles = triangles.flatten()
#     num_tri = len(triangles[0])
#     num_flat = len(flat_triangles)
#     i = 0

#     while i < num_tri:
#         triangle_indices = flat_triangles[i,i+num_tri,i+num_tri+num_tri]
#         # Get the vertices of the triangle
#         triangle_vertices = vertices[triangle_indices]

#         # Compute the axis-aligned bounding box
#         min_corner = np.min(triangle_vertices, axis=0)
#         max_corner = np.max(triangle_vertices, axis=0)

#         # Create a BoundingBox object and add it to the list
#         bounding_box = BoundingBox(min_corner, max_corner, triangle_indices)
#         bounding_boxes.append(bounding_box)
#         i += 1
#     return bounding_boxes

def construct_bounding_boxes(vertices, triangles):
    bounding_boxes = []

    for i in range(len(triangles[0])):
        # Extract indices for each vertex of the triangle
        triangle_indices = triangles[:, i].astype(int)

        # Get the vertices of the triangle
        triangle_vertices = vertices[:, triangle_indices]

        # Compute the axis-aligned bounding box
        min_corner = np.min(triangle_vertices, axis=0)
        max_corner = np.max(triangle_vertices, axis=0)

        # Create a BoundingBox object and add it to the list
        bounding_box = BoundingBox(min_corner, max_corner, triangle_indices)
        bounding_boxes.append(bounding_box)

    return bounding_boxes

def find_closest_point_bbox(point, kdtree, bounding_boxes, vertices):
    # Query the k-d tree for the nearest bounding box
    distances, indices = kdtree.query(point, k=1)
    nearest_box = bounding_boxes[indices]

    # Check the triangle within the nearest bounding box
    triangle_indices = nearest_box.triangle
    r_vertex_index = triangle_indices[0]
    p_vertex_index = triangle_indices[1]
    q_vertex_index = triangle_indices[2]

    r_coor = vertices[:, r_vertex_index]
    p_coor = vertices[:, p_vertex_index]
    q_coor = vertices[:, q_vertex_index]

    closest_point = find_closest_point_kd(point, r_coor, p_coor, q_coor)
    
    return closest_point

def find_closest_point_kd(point, r, p, q):
    """
    Find the closest point on the mesh surface defined by vertices and triangles to the given point.
    
    Parameters:
    point: Given one point as a NumPy array.
    r, p, q: Coordinates of vertices as a NumPy array.

    Returns:
    minpoint: the closest point to the given point on the surface mesh
    """
    c_ij = np.zeros([3, 1]) # closest point initialize with zeros 
    S = np.zeros([3, 2])

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

    return c_star


# def find_closest_point_bbox(point, kdtree, bounding_boxes, vertices):
#     # Query the k-d tree for the nearest bounding box
#     distances, indices = kdtree.query(point, k=1)
#     nearest_box = bounding_boxes[indices]

#     # Check the triangle within the nearest bounding box
#     triangle_indices = nearest_box.triangle
#     r_vertex_index = triangle_indices[0]
#     p_vertex_index = triangle_indices[1]
#     q_vertex_index = triangle_indices[2]

#     r_coor = vertices[:, r_vertex_index]
#     p_coor = vertices[:, p_vertex_index]
#     q_coor = vertices[:, q_vertex_index]

#     ver = []
#     ver.append(r_coor)
#     ver.append(p_coor)
#     ver.append(q_coor)

#     closest_point = find_closest_point_kd(point, ver, triangle_indices)
    
#     return closest_point

# def find_closest_point_kd(point, vertices, triangle):
#     """
#     Find the closest point on the mesh surface defined by vertices and triangles to the given point.
    
#     Parameters:
#     point: Given one point as a NumPy array.
#     vertices: Coordinates of vertices as a NumPy array.
#     triangles: Indices of vertex coordinates for each triangle as a NumPy array.

#     Returns:
#     minpoint: the closest point to the given point on the surface mesh
#     """
#     r = vertices[0]
#     p = vertices[1]
#     q = vertices[2]
    
#     c_ij = np.zeros([3, 1]) # closest point initialize with zeros 
#     S = np.zeros([3, 2])

#     for j in range(3):
#             S[j][0] = q[j] - p[j]
#             S[j][1] = r[j] - p[j]
#     b = point - p
#     soln = la.lstsq(S, b, rcond=None)
#     l = soln[0][0]
#     m = soln[0][1]

#     mid = p + l * (q - p) + m * (r - p)

#     if l >= 0 and m >= 0 and l + m <= 1:
#         c_star = mid
#     elif l < 0:
#         c_star = project_on_segment(mid , r, p)
#     elif m < 0:
#         c_star = project_on_segment(mid , p, q)
#     else:  # l + m > 1
#         c_star = project_on_segment(mid , q, r)

#     return c_star
