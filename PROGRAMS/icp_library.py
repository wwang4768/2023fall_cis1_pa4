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

def calc_difference(point_set_a, point_set_b):
    """
    Calculates the Euclidean distance between corresponding points in two point clouds.

    Parameters:
    point_set_a: point cloud a 
    point_set_b: point cloud b 

    Returns: 
    1D array with distances between each pair of corresponding points.
    """
    dist = np.zeros(np.shape(point_set_a)[0])
    for i in range(np.shape(point_set_a)[0]):
        dist[i] = np.linalg.norm(point_set_b[i] - point_set_a[i])

    return dist

def transform_tip_positions(tip_positions, frame_transformation):
    """
    Transforms tip positions with the given frame transformation. The func. will be useful in PA#4, In PA#3 we assume F_reg is I.
    
    Parameters:
    tip_positions: Array containing positions of tip with respect to rigid body B
    frame_transformation: Frame transformation

    Returns: 
    transformed_tip_positions: Transformed array of points

    """
    transformed_tip_pos = []
    for i in range(len(tip_positions)):
        registration = cal.setRegistration()
        new = registration.apply_transformation_single_pt(tip_positions[i], frame_transformation)
        transformed_tip_pos.append(new)
    return transformed_tip_pos

def findClosestPoints(vertices, triangles, startPoints, searchMode, maxIterations):
    """
    Finds the closest points on a 3D mesh to a set of starting points and computes the registration frame using an iterative closest point algorithm.

    Parameters:
    vertices: 3D coordinates of the mesh vertices.
    triangles: Indices of the vertices that form each triangle of the mesh.
    startPoints: 3D coordinates of the points from which the closest points on the mesh are to be found.
    searchMode: The mode of search to be used, either 'kd' for k-d tree or any other string for a linear search.
    maxIterations: The maximum number of iterations to perform in the ICP algorithm.

        :type vertices: np.array, shape (3, N), where N is the number of vertices.
        :type triangles: np.array of int, shape (3, M), where M is the number of triangles.
        :type startPoints: np.array, shape (3, P), where P is the number of starting points.
        :type searchMode: str
        :type maxIterations: int`

    Returns: 
    A tuple containing the array of closest points on the mesh and the final registration frame matrix.
        :type: (np.array of np float with shape (4, 4))
    """
    registrationFrame = np.identity(4) # initial transformation assumption
    iteration = 0 
    previousError = collections.deque(maxlen=2)
    previousError.append(0)
    registration = cal.setRegistration()
    # bounding_boxes = construct_bounding_boxes(vertices, triangles)
    # box_centers = [(box.min_corner + box.max_corner) / 2 for box in bounding_boxes]
    # kdtree = KDTree(box_centers)
    vertex_list = [vertices[:, i] for i in range(vertices.shape[1])]
    kdtree = KDTree(vertex_list)

    while iteration < maxIterations:
        transformedPoints = transform_tip_positions(startPoints, registrationFrame)
        allClosestPoints = []

        for point in transformedPoints:
            if searchMode != 'kd':
                closestPoint = linear_search_closest_point(point, vertices, triangles)
            else:
                closestPoint = find_closest_point_vertex_kd(point, kdtree, vertices, triangles)
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

    Parameters:
    tolerance: Tolerance threshold for sum of squared differences.
    oldFrame: Previous frame transformation.
    newFrame: Current frame transformation.
    errorHistory: History of previous errors to assess convergence.

        :type tolerance: float
        :type oldFrame: 4x4 matrix
        :type newFrame: 4x4 matrix
        :type errorHistory: collections.deque

    Returns:
    Bool: Whether the difference is within the tolerance or not.
    """
    error = sum((oldFrame[:3, 3][i] - newFrame[:3, 3][i]) ** 2 +
                sum((oldFrame[:3, :3][i][j] - newFrame[:3, :3][i][j]) ** 2 for j in range(3))
                for i in range(3))

    if error < tolerance or (errorHistory and abs(error - errorHistory[0]) < tolerance):
        return True

    errorHistory.append(error)
    return False

def find_closest_point_kd(point, r, p, q):
    """
    Find the closest point on the mesh surface defined by vertices and triangles to the given point.
    
    Parameters:
    point: Given one point as a NumPy array.
    r, p, q: Coordinates of vertices as a NumPy array.

    Returns:
    minpoint: the closest point to the given point on the surface mesh
    """
    #c_ij = np.zeros([3, 1])
    c_star = np.zeros([3, 1]) # closest point initialize with zeros 

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

def find_closest_point_vertex_kd(point, kdtree, vertices, triangles):
    """
    Finds the closest point on a mesh to a given point using a k-d tree for initial vertex proximity and subsequent triangle refinement.

    Parameters:
    point: The 3D coordinates of the point from which the closest point on the mesh is sought.
    kdtree: A k-d tree data structure containing the mesh vertices, used to quickly find the nearest vertex to the given point.
    vertices: A 2D array of the mesh's vertices, where each column represents the x, y, z coordinates of a vertex.
    triangles: A 2D array representing the mesh's triangles, where each column has three indices that correspond to the vertices forming that triangle.

        :type point: np.array with shape (3,)
        :type kdtree: scipy.spatial.KDTree
        :type vertices: np.array with shape (3, N), N being the number of vertices
        :type triangles: np.array with shape (3, M), M being the number of triangles

    Returns:
    The coordinates of the closest point on the mesh to the given point.
    """
    # kd tree for the nearest vertex
    distances, indices = kdtree.query(point, k=1)
    nearest_vertex_index = indices
    nearest_vertex = vertices[:, nearest_vertex_index]

    # find triangles containing the nearest vertex
    triangles_containing_vertex = np.where(np.any(triangles == nearest_vertex_index, axis=0))[0]

    min_distance = float('inf')
    closest_point = None

    # iterate over triangles containing the nearest vertex
    for triangle_index in triangles_containing_vertex:

        r_vertex_index, p_vertex_index, q_vertex_index = triangles[:, triangle_index]
        r_coor = vertices[:, int(r_vertex_index)]
        p_coor = vertices[:, int(p_vertex_index)]
        q_coor = vertices[:, int(q_vertex_index)]

        # find the closest point on the current trianglar mesh
        current_closest_point = find_closest_point_kd(point, r_coor, p_coor, q_coor)
        current_distance = np.linalg.norm(point - current_closest_point)

        if current_distance < min_distance:
            min_distance = current_distance
            closest_point = current_closest_point

    return closest_point

'''
class BoundingBox:
    def __init__(self, min_corner, max_corner, triangle_vertex_indices):
        self.min_corner = np.array(min_corner)
        self.max_corner = np.array(max_corner)
        self.triangle = triangle_vertex_indices  

    def contains_point(self, point):
        return np.all(point >= self.min_corner) and np.all(point <= self.max_corner)

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
'''

