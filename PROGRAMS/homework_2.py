import numpy as np, copy, os, re, argparse
from calibration_library import *
from dataParsing_library import *
from debug_test import *
from distortion_library import * 

def main(): 
    # User interface prompt that takes input from user
    parser = argparse.ArgumentParser(description='homework_2 input')
    parser.add_argument('input_type', help='The debug or unknown input data to process')
    parser.add_argument('choose_set', help='The alphabetical index of the data set')
    args = parser.parse_args()

    # Read in input dataset
    script_directory = os.path.dirname(__file__)
    dirname = os.path.dirname(script_directory)
    base_path = os.path.join(dirname, f'PROGRAMS\\pa2_student_data\\pa2-{args.input_type}-{args.choose_set}')

    calbody = base_path + '-calbody.txt'
    calbody_point_cloud = parseData(calbody)
    d0, a0, c0 = parseCalbody(calbody_point_cloud)

    calreading = base_path + '-calreadings.txt'
    calreading_point_cloud = parseData(calreading)
    calreading_frames = parseFrame(calreading_point_cloud, 8+8+27) # 8 optical markers on calibration object and 27 EM markers on calibration object
    
    empivot = base_path + '-empivot.txt'
    empivot_point_cloud = parseData(empivot)
    empivot_frames = parseFrame(empivot_point_cloud, 6) # stores the list of 12 frames, each of which contains data of 6 EM markers on probe 
    
    optpivot = base_path + '-optpivot.txt'
    optpivot_point_cloud = parseData(optpivot) # stores the list of 12 frames, each of which contains data of 8 optical markers on EM base & 6 EM markers on probe
    optpivot_em_frames, optpivot_opt_frames = parseOptpivot(optpivot_point_cloud, 8, 6) 

    # new inputs specific to PA2 to be incorporated 
    ct_fid = base_path + '-ct-fiducials.txt'
    ct_fid_point_cloud = parseData(ct_fid)

    em_fid = base_path + '-em-fiducialss.txt'
    em_fid_point_cloud = parseData(em_fid)
    em_fid_frames = parseFrame(em_fid_point_cloud, 6) # stores the list of 6 frames, each of which contains data of 6 EM markers on probe 


    em_nav = base_path + '-EM-nav.txt'
    em_nav_point_cloud = parseData(em_nav)
    em_nav_frames = parseFrame(em_nav_point_cloud, 6) # stores the list of 4 frames, each of which contains data of 6 EM markers on probe 

    registration = setRegistration()
    np.set_printoptions(formatter={'float': '{:.2f}'.format})

    # Step 1
    # Use transformed Ci expected from ci, compared against real Ci in calreading to calculate degree of distortion
    source_points_d = d0
    trans_matrix_d = []
    target_points = []

    for i in range(125):
        target_points = calreading_frames[i][:8]
        transformation_matrix = registration.calculate_3d_transformation(source_points_d, target_points)
        trans_matrix_d.append(transformation_matrix)

    source_points_a = a0
    trans_matrix_a = []
    target_points = []

    for i in range(125):
        target_points = calreading_frames[i][8:16]
        transformation_matrix = registration.calculate_3d_transformation(source_points_a, target_points)
        trans_matrix_a.append(transformation_matrix)
    
    source_points_c = c0
    transformation_matrix = []
    transformed_point = []
    distorted_data = []

    for i in range(125):
        distorted_data.append(calreading_frames[i][16:43])
        transformation_matrix = np.dot(np.linalg.inv(trans_matrix_d[i]), trans_matrix_a[i])
        transformed_point.append(registration.apply_transformation(source_points_c, transformation_matrix))
    
    # transformed_point should contain 3375 = 27 * 125 frames Ci expected 

    # Step 2
    # undistort empivot_frames
    # distorted_data
    ground_truth_data = transformed_point
    calibrator_corrected = DewarpingCalibrationCorrected()
    sample_data = empivot_frames

    distorted_data_comb = []
    ground_truth_data_comb = []

    for i in range(125):
        for j in range(27):
            distorted_data_comb.append(distorted_data[i][j])
            ground_truth_data_comb.append(ground_truth_data[i][j])

    # 125 frames, each of which has 27 points 
    calibrator_corrected.fit(distorted_data_comb, ground_truth_data_comb)

    # 12 frames, each of which has 6 points
    # Everything related G has to be corrected to dewarp the distortion
    corrected_empivot_sample = []
    for i in range(12):
        corrected_empivot_sample.append(calibrator_corrected.correction(sample_data[i]))

    # Step 3
    empivot_frames = corrected_empivot_sample
    # Initalize the set for gj = Gj - G0
    translated_points_Gj = copy.deepcopy(empivot_frames)
    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    midpoint = np.mean(empivot_frames, axis=1)
    trans_matrix_FG = []

    for i in range(12):
        for j in range(6):
            p = empivot_frames[i][j] - midpoint[i]
            translated_points_Gj[i][j] = p
    
    # fix gj as the original starting positions
    source_points = translated_points_Gj[0]
    for i in range(12):
        target_points = empivot_frames[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points, target_points)
        trans_matrix_FG.append(transformation_matrix)

    p_tip_G, _ = registration.pivot_calibration(trans_matrix_FG)

    # Step 4
    # Initalize the set for gj = Gj - G0
    # undistort em_fid_frames
    corrected_em_fid_sample = []
    for i in range(6):
        corrected_em_fid_sample.append(calibrator_corrected.correction(em_fid_frames[i]))

    translated_points_Gj_fid = copy.deepcopy(corrected_em_fid_sample)
    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    trans_matrix_FG_fid = []
    
    # fix gj as the original starting positions
    #source_points_fid = translated_points_Gj_fid[0]
    source_points_fid = translated_points_Gj[0]
    for i in range(6):
        target_points_fid = corrected_em_fid_sample[i]
        transformation_matrix_fid = registration.calculate_3d_transformation(source_points_fid, target_points_fid)
        trans_matrix_FG_fid.append(transformation_matrix_fid)

    pivot_Bj = []
    # may have to loop through - apply transformation matrices to p_pivot_G
    for i in range(6):
        pivot_Bj.append(registration.apply_transformation_single_pt(p_tip_G, trans_matrix_FG_fid[i]))    

    # Step 5
    source_points_bj = ct_fid_point_cloud #bj 
    target_points_bj = pivot_Bj
    transformation_matrix_Bj = registration.calculate_3d_transformation(source_points_bj, target_points_bj)

    # Step 6
    # Initalize the set for gj = Gj - G0
     # undistort em_nav_frames
    corrected_em_nav_sample = []
    for i in range(4):
        corrected_em_nav_sample.append(calibrator_corrected.correction(em_nav_frames[i]))

    translated_points_Gj_nav = copy.deepcopy(corrected_em_nav_sample)

    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    trans_matrix_FG_nav = []

    # fix gj as the original starting positions
    source_points_nav = translated_points_Gj[0]
    for i in range(4):
        target_points_nav = corrected_em_nav_sample[i]
        transformation_matrix_nav = registration.calculate_3d_transformation(source_points_nav, target_points_nav)
        trans_matrix_FG_nav.append(transformation_matrix_nav)
    # p_dimple * FGi * Freg 

    output = []
    for i in range(4):
        trans_FG = registration.apply_transformation_single_pt(p_tip_G, trans_matrix_FG_nav[i])
        output.append(trans_FG)
    
    for i in range(4):
        output[i] = registration.apply_transformation_single_pt(output[i], np.linalg.inv(transformation_matrix_Bj))

    # format output
    output_name = f'pa2-{args.input_type}-{args.choose_set}-output2.txt'

    max_length = max(max(len(f"{point[0]:.2f}"), len(f"{point[1]:.2f}"), len(f"{point[2]:.2f}")) for point in output)

    with open(output_name, "w") as file:
        file.write('4, ' + output_name + '\n')

        for point in output:
            formatted_point = '  ' + f"{ point[0]:>{max_length}.2f}, {point[1]:>{max_length}.2f}, {point[2]:>{max_length}.2f}\n"
            file.write(formatted_point)     
 
if __name__ == "__main__":
    main()
    
    # v = validate()
    # file1 = 'pa1_student_data\PA1 Student Data\pa1-debug-g-output1.txt'
    # file2 = 'OUTPUT\pa1-unknown-g-output.txt'
    
    # percentage_differences = v.calculate_error_from_sample(file1, file2, use_reference=0)
    # print(np.mean(percentage_differences))
    
    