import unittest
import numpy 
from calibration_library import *
from dataParsing_library import *
from distortion_library import *
import icp_library as icp
import collections
import numpy.testing as npt
from scipy.spatial import KDTree

class TestDistortionCorrection(unittest.TestCase):

    def setUp(self):
        # Initialize class data here
        np.random.seed(0) 
        self.source_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.target_points = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
        self.set_registration = setRegistration()
        vertices = np.array([[0.0, 0.0, 0.0],
                             [2.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0]])
        triangles = np.array([[0, 1, 2],
                      [1, 2, 3],
                      [2, 3, 4]])

        self.kdtree = KDTree(vertices)
        self.vertices = vertices
        self.triangles = triangles
        
    def test_linear_search_closest_point(self):
        # Test 1
        vertices = np.array([[0, 2, 0],
                        [1, 2, 3],
                        [0, 0, 0]], dtype=np.float64)
        to_add = np.array([[4, 4, 4],
                        [0, 0, 0],
                        [0, 0, 0]], dtype=np.float64)
        for i in range(2):
            vertices = np.hstack((vertices, vertices[:, (-3, -2, -1)] + to_add))
        triangle_indices = np.array([[0, 0, 1, 1],
                            [1, 1, 2, 3],
                            [2, 3, 5, 5]], dtype=int)
        triangle_indices = np.hstack((triangle_indices, triangle_indices + 3 * np.ones((3, 4), dtype=int), np.array([[6], [7], [8]],dtype=int)))
        s = vertices.copy()
        s[-1, :] += 4
        c_calc = np.zeros([np.shape(vertices)[1],3])
        for i in range(np.shape(s)[1]):
            c_calc[i, :] = icp.linear_search_closest_point(s[:, i], vertices, triangle_indices)
        assert np.all(np.abs(vertices - c_calc.T) <= 1e-3)

        # Test 2
        vertices = np.array([[0, 0, 4],
                            [1, 3, 2],
                            [0, 0, 0]], dtype=np.float64)
        vertices_index = np.array([[0],
                        [1],
                        [2]], dtype=int)

        # Case 1 - point in triangle, not in plane
        s = np.array([2, 2, 2])
        c_2 = np.array([2, 2, 0])
        c_calc = icp.linear_search_closest_point(s, vertices, vertices_index)
        assert np.all(np.abs(c_2 - c_calc) <= 1e-3)

        # Case 2 - point not in triangle, not in plane
        s = np.array([5, 2, 6])
        c_3 = np.array([4, 2, 0])
        c_calc = icp.linear_search_closest_point(s, vertices, vertices_index)
        assert np.all(np.abs(c_3 - c_calc) <= 1e-3)

        # Case 3 - point in triangle, not in plane
        s = np.array([2.5, 2, 0])
        c_3 = np.array([2.5, 2, 0])
        c_calc = icp.linear_search_closest_point(s, vertices, vertices_index)
        assert np.all(np.abs(c_3 - c_calc) <= 1e-3)

        # Case 4 - point not in triangle, in plane
        s = np.array([-2, 1.5, 0])
        c_4 = np.array([0, 1.5, 0])
        c_calc = icp.linear_search_closest_point(s, vertices, vertices_index)
        assert np.all(np.abs(c_4 - c_calc) <= 1e-3)

    def test_project_on_segment(self):
        p = np.array([0, 10, 0])
        q = np.array([0, 0, 0])
        # c is to the left or right of the segment in the plane
        c_1 = np.array([-2, 7, 0])
        c_exp = np.array([0, 7, 0])
        c_star = icp.project_on_segment(c_1, p, q)
        assert np.all(np.abs(c_exp - c_star) <= 1e-3)
        
        # c is on the segment in the plane
        c_2 = np.array([0, 11, 0])
        c_exp = np.array([0, 10, 0])
        c_star = icp.project_on_segment(c_2, p, q)
        assert np.all(np.abs(c_exp - c_star) <= 1e-3)

    def test_calculate_3d_transformation(self):
        transformation_matrix = self.set_registration.calculate_3d_transformation(self.source_points, self.target_points)
        self.assertEqual(transformation_matrix.shape, (4, 4))

    def test_apply_transformation(self):
        transformation_matrix = self.set_registration.calculate_3d_transformation(self.source_points, self.target_points)
        transformed_points = self.set_registration.apply_transformation(self.source_points, transformation_matrix)
        self.assertEqual(transformed_points.shape, self.source_points.shape)
    
    def test_transform_tip_positions(self):
        tip_positions = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        frame_transformation = np.identity(4)
        result = icp.transform_tip_positions(tip_positions, frame_transformation)
        self.assertEqual(len(result), len(tip_positions))

    def test_findClosestPoints(self):
        vertices = np.array([[0, 2, 0], [1, 2, 3], [0, 0, 0]], dtype=np.float64)
        to_add = np.array([[4, 4, 4], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        for i in range(2):
            vertices = np.hstack((vertices, vertices[:, (-3, -2, -1)] + to_add))
        triangle_indices = np.array([[0, 0, 1, 1], [1, 1, 2, 3], [2, 3, 5, 5]], dtype=int)
        triangle_indices = np.hstack((triangle_indices, triangle_indices + 3 * np.ones((3, 4), dtype=int), np.array([[6], [7], [8]], dtype=int)))

        startPoints = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        searchMode = 'kd'

        closest_points, registration_frame = icp.findClosestPoints(vertices, triangle_indices, startPoints, searchMode)

        expected_closest_points = [
            np.array([0.0, 2.00688557, 0.0]),
            np.array([4.9905831, 2.00412785, 0.0]),
            np.array([10.0, 2.0, 0.0])
        ]

        ifValid = False
        for expected, actual in zip(expected_closest_points, closest_points):
            for e, a in zip(expected, actual):
                ifValid = np.isclose(a, e, rtol=1e-1)
                if ifValid == False:
                    break
        self.assertTrue(ifValid)

    def test_hasConverged(self):
        tolerance = 1e-4
        oldFrame = np.identity(4)
        newFrame_converged = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        newFrame_not_converged = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        errorHistory = collections.deque(maxlen=2)
        errorHistory.append(0)

        # Test converged scenario
        result_converged = icp.hasConverged(tolerance, oldFrame, newFrame_converged, errorHistory)
        self.assertTrue(result_converged)

        # Test not converged scenario
        result_not_converged = icp.hasConverged(tolerance, oldFrame, newFrame_not_converged, errorHistory)
        self.assertFalse(result_not_converged)

        # Test with small difference
        newFrame_small_difference = newFrame_converged + 1e-5
        result_small_difference = icp.hasConverged(tolerance, oldFrame, newFrame_small_difference, errorHistory)
        self.assertTrue(result_small_difference)

    def test_find_closest_point_kd_on_triangle(self):
        # Test on triangle 
        point = np.array([1.0, 1.0, 1.0])
        r = np.array([0.0, 0.0, 0.0])
        p = np.array([2.0, 0.0, 0.0])
        q = np.array([0.0, 2.0, 0.0])

        result = icp.find_closest_point_kd(point, r, p, q)

        expected_result = np.array([1.0, 1.0, 0.0])
        npt.assert_allclose(result, expected_result, rtol=1e-3)

    def test_find_closest_point_kd_outside_triangle(self):
        # Test outside triangle 
        point = np.array([3.0, 3.0, 3.0])
        r = np.array([0.0, 0.0, 0.0])
        p = np.array([2.0, 0.0, 0.0])
        q = np.array([0.0, 2.0, 0.0])

        result = icp.find_closest_point_kd(point, r, p, q)

        expected_result = np.array([1.0, 1.0, 0.0])
        npt.assert_allclose(result, expected_result, rtol=1e-3)


    def test_parseFrame(self):
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        frame_chunk = 4
        frames = parseFrame(test_data, frame_chunk)
        self.assertEqual(len(frames), len(test_data) // frame_chunk)

if __name__ == '__main__':
    unittest.main()

'''
#Test parsing
def test_parseMesh(self):
    input_file = '2023_pa345_student_data\\Problem3Mesh.sur'
    vertices_num = 1568

    expected_vertices = np.array([-23.786148, -16.420282, -48.229988])
    expected_triangles = np.array([12, 19, 1])

    vertices_cloud, triangles_cloud = parseMesh(input_file, vertices_num)

    np.testing.assert_equal(vertices_cloud[0], expected_vertices)
    np.testing.assert_equal(triangles_cloud[0], expected_triangles)

    
def test_find_closest_point_vertex_kd_on_vertex(self):
    point = np.array([0.0, 0.0, 0.0])

    result = icp.find_closest_point_vertex_kd(point, self.kdtree, self.vertices, self.triangles)

    expected_result = np.array([0.0, 0.0, 0.0])
    npt.assert_allclose(result, expected_result, rtol=1e-1)

def test_find_closest_point_vertex_kd_outside_triangle(self):
    point = np.array([3.0, 3.0, 3.0])

    result = icp.find_closest_point_vertex_kd(point, self.kdtree, self.vertices, self.triangles)

    expected_result = np.array([2.0, 0.0, 0.0])
    npt.assert_allclose(result, expected_result, rtol=1e-1)
'''