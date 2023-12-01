import numpy as np
import re

def parseData(input_file):
    points = [] 

    with open(input_file, 'r') as file:
        # skip the first line
        next(file)

        for line in file:
            #x, y, z = map(float, line.strip().split(','))
            # Split based on any whitespace character
            #coordinates = line.strip().split()
            coordinates = re.split(r'[,\s]+', line.strip())

            # Convert each coordinate to float
            x, y, z = map(float, coordinates)
            point = (x, y, z)
            points.append(point)

    point_cloud = np.array(points)
    return point_cloud

def parseCalbody(point_cloud):
    # Number of optical markers on EM base
    d = point_cloud[:8]
    # number of optical markers on calibration object
    a = point_cloud[8:16]
    # number EM markers on calibration object
    c = point_cloud[-27:]
    return d, a, c

def parseMesh(input_file, vertices_num):
    vertices = []
    triangles = []
    count = 0 

    with open(input_file, 'r') as file:
        # skip the first line
        next(file)

        for line in file:
            #x, y, z = map(float, line.strip().split(','))
            # Split based on any whitespace character
            if count < vertices_num:
                coordinates = line.strip().split()
                x, y, z = map(float, coordinates)
                point = (x, y, z)
                vertices.append(point)
                count += 1
            elif count == vertices_num:
                count +=1
            else:
                coordinates = line.strip().split()
                x, y, z, _, _, _ = map(float, coordinates)
                point = (x, y, z)
                triangles.append(point)
                count += 1

    vertices_cloud = np.array(vertices)
    triangles_cloud = np.array(triangles)
    return vertices_cloud, triangles_cloud

def parseOptpivot(point_cloud, len_chunk_d, len_chunk_h):
    frames_d = []
    frames_h = []

    chunk_size_d = len_chunk_d
    chunk_size_h = len_chunk_h

    current_list = 'D'
    temp = []

    for p in point_cloud:
        temp.append(p)
        
        if len(temp) == chunk_size_d and current_list == 'D':
            frames_d.append(temp)
            temp = []
            current_list = 'H'
        elif len(temp) == chunk_size_h and current_list == 'H':
            frames_h.append(temp)
            temp = []
            current_list = 'D'
    return frames_d, frames_h

def parseFrame(point_cloud, frame_chunk):
    frames = []
    for i in range(0, len(point_cloud), frame_chunk):
        row = point_cloud[i:i+frame_chunk]
        frames.append(row)
    return frames