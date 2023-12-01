import numpy as np
from scipy.special import comb

# Create an instance of the DewarpingCalibrationCorrected class, fit it to our data, and correct a sample point
class DewarpingCalibrationCorrected:

# Initialize the script in the class 
    def __init__(self, degree=5):
        self.degree = degree
        self.coefficients = None
        self.q_min = None
        self.q_max = None
    '''
    The function scales input data with upper and lower bounds to shape the data into [0,1] 

    INPUT:
    q_min: minimum values used for scaling (Numpy Array)
    q_max: maximum values used for scaling (Numpy Array)
    
    OUTPUT:
    NumPy array of scaled data
    '''  
    @staticmethod
    def scale_to_box(data, q_min, q_max):
        return (data - q_min) / (q_max - q_min)

    '''
    The function constructs Berstein polynomial

    INPUT:
    N: degree of the Bernstein polynomial
    k: Bernstein polynomial parameter
    u: input value for the Bernstein polynomial

    OUTPUT:
    Computed Berstein polynomial
    '''
    @staticmethod
    def bernstein(N, k, u):
        return comb(N, k) * (1 - u) ** (N - k) * u ** k
    '''
    The function constructs F matrix
    
    INPUT:
    degree: The degree of the Bernstein polynomial.
    u: input values for the polynomial (NumPy Array)
    
    OUTPUT:
    Computed F matrix
    '''
    def build_f_matrix(self, u):
        n_points = len(u)
        f_mat = np.zeros([n_points, (self.degree + 1) ** 3])

        for n in range(n_points):
            c = 0
            for i in range(self.degree + 1):
                for j in range(self.degree + 1):
                    for k in range(self.degree + 1):
                        f_mat[n][c] = self.bernstein(self.degree, i, u[n][0]) * \
                                      self.bernstein(self.degree, j, u[n][1]) * \
                                      self.bernstein(self.degree, k, u[n][2])
                        c += 1
        return f_mat

    '''
    The function takes the distorted calibration data and computes the coefficients of the distortion correction model using a least squares optimization 
    
    INPUT:
    distorted_data: input data set that needs to be corrected 
    ground_truth: expected data to be fitted 
    
    OUTPUT:
    Computed F matrix
    '''
    def fit(self, distorted_data, ground_truth):
        self.q_min = np.min(distorted_data, axis=0)
        self.q_max = np.max(distorted_data, axis=0)

        normalized_data = DewarpingCalibrationCorrected.scale_to_box(distorted_data, self.q_min, self.q_max)
        F = self.build_f_matrix(normalized_data)

        # apply least square
        self.coefficients = np.linalg.lstsq(F, ground_truth, rcond=None)[0]

    '''
    The function applies the distortion correction to input data

    INPUT:
    data: data to be corrected (Numpy Array)
    q_min: minimum values used for scaling (Numpy Array)
    q_max: maximum values used for scaling (Numpy Array)
    coefficients: A NumPy array of coefficients obtained from calibration

    OUTPUT:
    Corrected data after applying F matrix and coefficients
    '''
    def correction(self, data):

        # check if valid input
        if self.coefficients is None:
            raise ValueError("model is not fitted yet - invalid coefficients")

        normalized_data = DewarpingCalibrationCorrected.scale_to_box(data, self.q_min, self.q_max)
        F = self.build_f_matrix(normalized_data)

        # apply F matrix 
        corrected_data = F @ self.coefficients
        # print("corrected", corrected_data)
 
        return corrected_data

if __name__ == "__main__":
    distorted_data = np.random.rand(10000, 3) * 10
    ground_truth_data = distorted_data + np.random.randn(10000, 3) * 0.1
    calibrator_corrected = DewarpingCalibrationCorrected()
    sample_data = np.array([[6, 6, 6], [1,1,1], [3,3,3]])
    calibrator_corrected.fit(distorted_data, ground_truth_data)
    corrected_sample = calibrator_corrected.correction(sample_data)
    #print(corrected_sample)



