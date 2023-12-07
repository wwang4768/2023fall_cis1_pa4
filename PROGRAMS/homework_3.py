import numpy as np, copy, os, re, argparse
from calibration_library import *
from dataParsing_library import *
from debug_test import *
from distortion_library import * 
import icp_library as icp 

def main(): 
    # User interface prompt that takes input from user
    parser = argparse.ArgumentParser(description='homework_3 input')
    parser.add_argument('choose_set', help='The alphabetical index of the data set')
    parser.add_argument('input_type', help='The debug or unknown input data to process')
    args = parser.parse_args()

    # Read in input dataset
    script_directory = os.path.dirname(__file__)
    dirname = os.path.dirname(script_directory)
    base_path = os.path.join(dirname, f'PROGRAMS\\2023_pa345_student_data\\PA3-{args.choose_set}-{args.input_type}') 
    
    #Prolem3-BodyA.txt - 6 markers on Frame A and 1 tip 
    PA3_BodyA = os.path.join(dirname, f'PROGRAMS\\2023_pa345_student_data\\Problem3-BodyA.txt')
    PA3_BodyA_point_cloud = parseData(PA3_BodyA)

    #Prolem3-BodyB.txt - 6 markers on Frame B and 1 tip
    PA3_BodyB = os.path.join(dirname, f'PROGRAMS\\2023_pa345_student_data\\Problem3-BodyB.txt')
    PA3_BodyB_point_cloud = parseData(PA3_BodyB)

    #Problem3Mesh.sur - 1568 vertices and 3135 triangles (3 vertices index denoted as P Q R)
    # parse two data sets; for the second data set, keep first 3 datapoints
    PA3_Mesh = os.path.join(dirname, f'PROGRAMS\\2023_pa345_student_data\\Problem3Mesh.sur')
    PA3_vertices, PA3_triangles  = parseMesh(PA3_Mesh, 1568)

    #SampleReadingsTest
    SampleReading = base_path + '-SampleReadingsTest.txt'
    SampleReading_point_cloud = parseData(SampleReading)
    SampleReading_frames = parseFrame(SampleReading_point_cloud, 6+6+4) # 15 frames of 16 points, ignore last 4 for PA3

    registration = setRegistration()
    np.set_printoptions(formatter={'float': '{:.2f}'.format})

    """
    Step 1 - find tip position in relation to Frame B 
    
    find_tip_positions()
    a_frames / b_frames = 15 frames * 6 points - 1 set 
    led_a / led_b = PA3_BodyA first 6 points 
    tip_a = PA3_Body last point
    """
    # return 15 tip_position_b (aka d_k points)
    a_frames_set = []
    b_frames_set = []

    for frame in SampleReading_frames:
        a_frames_set.append(np.array(frame[:6]))
        b_frames_set.append(np.array(frame[6:12]))

    led_a = PA3_BodyA_point_cloud[:6]
    led_b = PA3_BodyB_point_cloud[:6]
    tip_a = PA3_BodyA_point_cloud[6]
    
    tip_pos = []

    # no need to concatenate; just use by frame sets 
    for i in range(len(a_frames_set)):
        registration_a = registration.calculate_3d_transformation(led_a, a_frames_set[i])
        registration_b = registration.calculate_3d_transformation(led_b, b_frames_set[i])
        registration_b = np.linalg.inv(registration_b)

        # perform matrix multiplication with the inverse
        combined_registration = np.matmul(registration_b, registration_a)
        transformed_tip_a = registration.apply_transformation_single_pt(tip_a, combined_registration)

        # convert 1D arrays to 2D arrays
        two_d = transformed_tip_a[:, np.newaxis]
        # transpose the tip position - tip_pos is a list and doesn't have a shape
        tip_pos.append(two_d)
        
    d_k = np.concatenate(tip_pos, axis=1)

    """
    Step 2 - return 15 closest point to c_k points 
    
    point = 15_tip_position_b from Step 1 
    vertices = 1568 vertices - need to transpose row = 3 column = 3136
    triangles = 3135 triangles' vertices index - need to transpose row = 3 column = 3136
    """
    vertices_trans = np.transpose(PA3_vertices)
    triangles_trans = np.transpose(PA3_triangles)
    c_k = []
    for i in range(len(a_frames_set)):
        pt = icp.linear_search_closest_point(d_k[:, i], vertices_trans, triangles_trans)
        two_d = pt[:, np.newaxis]
        c_k.append(two_d)
    closest_pt = np.concatenate(c_k, axis=1)
    
    """
    Step 3 
    put in returns from step 1 and step 2
    return 15 distance 
    """
    distance = icp.calc_difference(c_k, tip_pos)

    # format Output
    output_name = f'PA3-{args.choose_set}-{args.input_type}-Output.txt'

    # Initialize the output list
    output = []
    for i in range(len(a_frames_set)):
        # Transpose the data
        dk = np.transpose(tip_pos[i])
        ck = np.transpose(c_k[i])

        # initialize a row with a single space, then extend with dk, another single space as a placeholder, and ck
        row = []
        row.append(" ") 
        row.extend(dk.ravel())
        row.append("    ")
        row.extend(ck.ravel())

        row.append(round(distance[i], 3))
        output.append(row)

    max_length = max(max(len(f"{point:.2f}") for point in row if isinstance(point, float)) for row in output)

    # Write to the file
    with open(output_name, "w") as file:
        file.write(f'15 {output_name} 0\n')

        for row in output:
            formatted_row = ' '.join(
                f"{point:>{max_length}.2f} " if isinstance(point, float) else point for point in row[:-1])
            formatted_row += f"{row[-1]:>{max_length + 3}.3f}"
            file.write(formatted_row + '\n')
 
if __name__ == "__main__":
    main()
    
    # v = validate()
    # file1 = 'pa1_student_data\PA1 Student Data\pa1-debug-g-output1.txt'
    # file2 = 'OUTPUT\pa1-unknown-g-output.txt'
    
    # percentage_differences = v.calculate_error_from_sample(file1, file2, use_reference=0)
    # print(np.mean(percentage_differences))
    
    