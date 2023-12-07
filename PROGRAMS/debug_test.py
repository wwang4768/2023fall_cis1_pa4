import unittest
import numpy 
from calibration_library import *
from dataParsing_library import *
from distortion_library import *
import icp_library as icp

class TestDistortionCorrection(unittest.TestCase):

    def setUp(self):
        # Initialize class data here
        np.random.seed(0) 
        self.source_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.target_points = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
        self.set_registration = setRegistration()
        # self.distorted_data = np.random.rand(10000, 3) * 10
        # self.ground_truth_data = self.distorted_data + np.random.randn(10000, 3) * 0.1
        # self.sample_data = np.array([[6, 6, 6], [1,1,1], [3,0,3]])
        
    def test_find_closest_point(self):
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
    
    # Test parsing
    def test_parseMesh(self):
        input_file = '2023_pa345_student_data\\Problem3Mesh.sur'
        vertices_num = 1568

        expected_vertices = np.array([-23.786148, -16.420282, -48.229988])
        expected_triangles = np.array([12, 19, 1])

        vertices_cloud, triangles_cloud = parseMesh(input_file, vertices_num)

        np.testing.assert_equal(vertices_cloud[0], expected_vertices)
        np.testing.assert_equal(triangles_cloud[0], expected_triangles)

    def test_parseFrame(self):
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        frame_chunk = 4
        frames = parseFrame(test_data, frame_chunk)
        self.assertEqual(len(frames), len(test_data) // frame_chunk)

if __name__ == '__main__':
    unittest.main()

'''
def test_calc_difference(self):
        c_k_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        d_k_points = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        result = icp.calc_difference(c_k_points, d_k_points)
        expected_result = np.array([9.0, 8.660, 7.810]) 
        print(result)
        np.testing.assert_allclose(result, expected_result, atol=1e-3)

    def test_calibration_and_correction(self):
        calibrator_corrected = DewarpingCalibrationCorrected()
        calibrator_corrected.fit(self.distorted_data, self.ground_truth_data)
        corrected_sample = calibrator_corrected.correction(self.sample_data)

        # You can add assertions to verify that the corrected_sample is as expected
        self.assertTrue(np.allclose(corrected_sample, self.sample_data, rtol=1e-1, atol=1e-1))
    
    def test_fit(self):
        # Test the fit method
        calibrator = DewarpingCalibrationCorrected()
        calibrator.fit(self.distorted_data, self.ground_truth_data)
        coefficients = calibrator.coefficients
        q_min = calibrator.q_min
        q_max = calibrator.q_max

        # Assert that coefficients, q_min, and q_max are not None
        self.assertIsNotNone(coefficients)
        self.assertIsNotNone(q_min)
        self.assertIsNotNone(q_max)
    
    def test_correction(self):
        # Test the correction method
        calibrator = DewarpingCalibrationCorrected()
        calibrator.fit(self.distorted_data, self.ground_truth_data)
        corrected_sample = calibrator.correction(self.sample_data)

        # Assert that the corrected_sample has the expected shape
        self.assertEqual(corrected_sample.shape, self.sample_data.shape)

        # Assert that the corrected_sample is not None
        self.assertIsNotNone(corrected_sample)
'''