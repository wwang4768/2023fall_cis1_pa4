import numpy as np
import icp_library as ICP

def testProjectOnSegment(tolerance=1e-4):
    """
    Tests projection of a point onto a line segment.

    """
    print('\nTesting projection of a point onto a line segment...')
    print('\nDefine line segment from p to q:\np =')
    p = np.array([0, 10, 0])
    print(p)
    q = np.array([0, 0, 0])
    print('\nq =')
    print(q)
    print('\nNow make a test c to project:')
    print('\nCase 1: c is to the left or right of the segment in the plane:')
    print('c =')
    c = np.array([-2, 7, 0])
    print(c)
    c_exp = np.array([0, 7, 0])
    print('Expected c =')
    print(c_exp)
    print('Calculated c =')
    c_star = ICP.project_on_segment(c, p, q)
    print(c_star)
    print('\nAre c_exp and c_calc within tolerance?')
    assert np.all(np.abs(c_exp - c_star) <= tolerance)
    print(np.all(np.abs(c_exp - c_star) <= tolerance))
    print('\nCase 2: c is on the segment in the plane:')
    c = np.array([0, 11, 0])
    print(c)
    c_exp = np.array([0, 10, 0])
    print('Expected c =')
    print(c_exp)
    print('Calculated c =')
    c_star = ICP.project_on_segment(c, p, q)
    print(c_star)
    print('\nAre c_exp and c_calc within tolerance?')
    assert np.all(np.abs(c_exp - c_star) <= tolerance)
    print(np.all(np.abs(c_exp - c_star) <= tolerance))


def testICP_findclosestpoint(tolerance=1e-4):
    """
    Tests ICP using a series of points queried against a generated mesh. Uses a brute-force linear search.

    """
    print('\nComplicated case: Closest points to a mesh of triangles, using slow linear search')
    v_coords = np.array([[0, 2, 0],
                    [1, 2, 3],
                    [0, 0, 0]], dtype=np.float64)
    to_add = np.array([[4, 4, 4],
                       [0, 0, 0],
                       [0, 0, 0]], dtype=np.float64)
    for i in range(2):
        v_coords = np.hstack((v_coords, v_coords[:, (-3, -2, -1)] + to_add))
    print('\nVertices:')
    print(v_coords)
    tri_inds = np.array([[0, 0, 1, 1],
                         [1, 1, 2, 3],
                         [2, 3, 5, 5]], dtype=int)
    tri_inds = np.hstack((tri_inds, tri_inds + 3 * np.ones((3, 4), dtype=int), np.array([[6], [7], [8]],dtype=int)))

    print('\nTriangle Indices:')
    print(tri_inds)

    s = v_coords.copy()
    s[-1, :] += 4

    print('\nPoints to search for:')
    print(s)

    print('\nExpected points:')
    print(v_coords)

    print('\nCalculated points:')
    c_calc = np.zeros([np.shape(v_coords)[1],3])
    for i in range(np.shape(s)[1]):
        c_calc[i, :] = ICP.linear_search_closest_point(s[:, i], v_coords, tri_inds)

    print(c_calc.T)
    print('\nMatch within tolerance?')
    assert np.all(np.abs(v_coords - c_calc.T) <= tolerance)
    print(np.all(np.abs(v_coords - c_calc.T) <= tolerance))

    print('\nTest again, with new points')
    print('\nPoints to search for:')
    s = np.zeros((3, 11))
    s[0, :] = np.arange(0, 11)
    s[1, :] = 2
    print(s)

    print('\nExpected points:')
    print(s)

    print('\nCalculated points:')
    c_calc = ICP.linear_search_closest_point(s, v_coords, tri_inds)
    print(c_calc.data)

    print('\nMatch within tolerance?')
    assert np.all(np.abs(s - c_calc.data) <= tolerance)
    print(np.all(np.abs(s - c_calc.data) <= tolerance))

    print('\nLinear ICP tests passed!')


def testFindClosestPoint(tolerance=1e-4):
    """
    Tests finding closest point on a mesh of triangles

    :param tolerance: Max allowed difference between result and prediction.

    :type tolerance: float

    :return: None
    """
    print('\nTesting FindClosestPoint...')
    print('\nSimple case: Closest point to a single triangle')
    v_coords = np.array([[0, 0, 4],
                        [1, 3, 2],
                        [0, 0, 0]], dtype=np.float64)
    print('\nVertices:')
    print(v_coords)
    v_inds = np.array([[0],
                       [1],
                       [2]], dtype=int)

    print('\Case 1: Point in region over triangle, out of plane')
    print('\nPoint to match: s =')
    s = np.array([2, 2, 2])
    print(s)

    print('Expected point: c =')
    c = np.array([2, 2, 0])
    print(c)

    print('Calculated point: c_calc =')
    c_calc = ICP.linear_search_closest_point(s, v_coords, v_inds)
    print(c_calc)

    print('\nMatch within tolerance?')
    assert np.all(np.abs(c - c_calc) <= tolerance)
    print(np.all(np.abs(c - c_calc) <= tolerance))

    print('\nCase 2: Point outside region over triangle, out of plane')
    print('\nPoint to match: s =')
    s = np.array([5, 2, 6])
    print(s)

    print('Expected point: c =')
    c = np.array([4, 2, 0])
    print(c)

    print('Calculated point: c_calc =')
    c_calc = ICP.linear_search_closest_point(s, v_coords, v_inds)
    print(c_calc)

    print('\nMatch within tolerance?')
    assert np.all(np.abs(c - c_calc) <= tolerance)
    print(np.all(np.abs(c - c_calc) <= tolerance))

    print('\nCase 3: Point inside triangle, in plane')
    print('\nPoint to match: s =')
    s = np.array([2.5, 2, 0])
    print(s)

    print('Expected point: c =')
    c = np.array([2.5, 2, 0])
    print(c)

    print('Calculated point: c_calc =')
    c_calc = ICP.linear_search_closest_point(s, v_coords, v_inds)
    print(c_calc)

    print('\nMatch within tolerance?')
    assert np.all(np.abs(c - c_calc) <= tolerance)
    print(np.all(np.abs(c - c_calc) <= tolerance))

    print('\nCase 4: Point outside triangle, in plane')
    print('\nPoint to match: s =')
    s = np.array([-2, 1.5, 0])
    print(s)

    print('Expected point: c =')
    c = np.array([0, 1.5, 0])
    print(c)

    print('Calculated point: c_calc =')
    c_calc = ICP.linear_search_closest_point(s, v_coords, v_inds)
    print(c_calc)

    print('\nMatch within tolerance?')
    assert np.all(np.abs(c - c_calc) <= tolerance)
    print(np.all(np.abs(c - c_calc) <= tolerance))

testProjectOnSegment()
testICP_findclosestpoint()
# testFindClosestPoint()