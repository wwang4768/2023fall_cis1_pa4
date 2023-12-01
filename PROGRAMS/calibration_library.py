import numpy as np
from scipy.optimize import least_squares

class setRegistration:
    def calculate_3d_transformation(self, source_points, target_points):
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        centered_source = source_points - source_centroid
        centered_target = target_points - target_centroid

        H = np.dot(centered_source.T, centered_target)

        # singular value decomposition (SVD)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = target_centroid - np.dot(R, source_centroid)

        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t

        return transformation_matrix

    def compute_error(self, source_points, target_points, transformation):
        transformed_source = self.apply_transformation(source_points, transformation)
        squared_distances = np.sum((transformed_source - target_points) ** 2)
        error = np.sqrt(squared_distances)

        return error

    def apply_transformation(self, points, transformation):
        # conver to homogeneous coordinates by adding a column of 1
        homogeneous_points = np.column_stack((points, np.ones((points.shape[0], 1))))
        #print(homogeneous_points.shape)
        
        transformed_points = np.dot(homogeneous_points, transformation.T)

        # check scale 
        normalized_points = transformed_points[:, :3] / transformed_points[:, 3, np.newaxis]

        return normalized_points
    
    def apply_transformation_single_pt(self, point, transformation_matrix):
        # To facilitate computation
        point = np.append(point, 1.0)
        
        transformed_point = np.dot(transformation_matrix, point)
        
        # Extract the transformed 3D coordinates
        transformed_coordinates = transformed_point[:3]
        return transformed_coordinates

    def optimization_heuristics(self, parameters, transformation_matrices):
        p_tip = parameters[:3].reshape(3, 1)
        p_pivot = parameters[3:].reshape(3, 1)

        num_frames = len(transformation_matrices)
        transformed_frames = np.zeros((num_frames, 3))

        for j in range(num_frames):
            R_j = transformation_matrices[j, :3, :3]
            error_matrix = np.hstack([R_j, -np.eye(3)])
            concatenated_points = np.vstack([p_tip, p_pivot])
            transformed_frames[j] = np.dot(error_matrix, concatenated_points).flatten()
         # error as the difference between transformed and -p_j
        error = transformed_frames + transformation_matrices[:, :3, 3]
        return error.flatten()

    def pivot_calibration(self, transformation_matrices):
        transformation_matrices = np.array(transformation_matrices)
        num_frames = len(transformation_matrices)
        # initialize parameters (p_tip/p_pivot) with an initial guess
        initial_guess = np.zeros(6)
        result = least_squares(self.optimization_heuristics, initial_guess, args=(transformation_matrices,))

        p_tip_solution = result.x[:3]
        p_pivot_solution = result.x[3:]

        return p_tip_solution, p_pivot_solution
'''
class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def translate(self, dx, dy, dz):
        self.x += dx
        self.y += dy
        self.z += dz

class Rotation3D:
    def __init__(self, zRot=0, yRot=0, xRot=0):
        self.zRot = zRot  
        self.pitch = yRot  
        self.xRot = xRot 

    def __str__(self):
        return f"Yaw: {self.zRot}, Pitch: {self.pitch}, Roll: {self.xRot}"

    def rotate(self, yaw, pitch, roll):
        self.zRot += yaw
        self.pitch += pitch
        self.xRot += roll

    def matrix(self):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(self.xRot), -np.sin(self.xRot)],
                        [0, np.sin(self.xRot), np.cos(self.xRot)]])

        R_y = np.array([[np.cos(self.pitch), 0, np.sin(self.pitch)],
                        [0, 1, 0],
                        [-np.sin(self.pitch), 0, np.cos(self.pitch)]])

        R_z = np.array([[np.cos(self.zRot), -np.sin(self.zRot), 0],
                        [np.sin(self.zRot), np.cos(self.zRot), 0],
                        [0, 0, 1]])

        return np.dot(R_z, np.dot(R_y, R_x))

class Frame3D:
    def __init__(self, origin=Point3D(0, 0, 0), rotation=Rotation3D()):
        self.origin = origin
        self.rotation = rotation

    def transform_point(self, point):
        rotated_point = np.dot(self.rotation.matrix(), np.array([point.x, point.y, point.z]))

        transformed_point = Point3D(rotated_point[0] + self.origin.x,
                                    rotated_point[1] + self.origin.y,
                                    rotated_point[2] + self.origin.z)
        return transformed_point
'''

