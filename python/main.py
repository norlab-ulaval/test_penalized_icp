import yaml
from svgpathtools import svg2paths
import numpy as np
from math import cos, sin
import glob
import os
import parse
import pandas as pd
from collections import namedtuple
import subprocess


def delete_vtk_artifacts():
    print('Deleting .vtk file in current folder...')
    files = glob.glob('./*.vtk')
    for f in files:
        os.remove(f)


def parse_vtk(vtk_filename):
    def read_until(lines, label):
        line = next(lines)
        while not line.startswith(label):
            line = next(lines)
        return line

    with open(vtk_filename) as f:
        lines = iter(f.readlines())

    line = read_until(lines, "POINTS")
    nb_points = parse.parse("POINTS {:d} float", line)[0]

    points = np.zeros((nb_points, 3))
    for i in range(0, nb_points):
        line = next(lines)
        x, y, z = (float(nbr) for nbr in line.split())
        points[i, 0] = x
        points[i, 1] = y
        points[i, 2] = z
    points = pd.DataFrame(points, columns=["x", "y", "z"])

    try:
        line = read_until(lines, "LINES")
        nb_segs, size = parse.parse("LINES {:d} {:d}", line)

        segments = np.zeros((nb_segs, 2), dtype=int)
        for i in range(0, nb_segs):
            line = next(lines)
            size, id_read, id_ref = (int(nbr) for nbr in line.split())
            segments[i, 0] = id_read
            segments[i, 1] = id_ref
        segments = pd.DataFrame(segments, columns=["id_read", "id_ref"])

    except StopIteration:
        return points, None  # There are no segments

    # Parse Scalars until end of file
    try:
        while True:
            line = read_until(lines, "SCALARS")
            scalar_label, size = parse.parse("SCALARS {} float {:d}", line)
            assert next(lines).startswith("LOOKUP_TABLE")

            # Parse scarlar's table
            scalars = np.zeros(nb_segs)
            for i in range(0, nb_segs):
                scalars[i] = float(next(lines))
            segments[scalar_label] = scalars
    except StopIteration:
        pass

    return points, segments


def generate_config(estimator):
    outlier = create_icp_outlier_filter(estimator.name, ratio=estimator.param).convert_to_dict()
    config = {
        'inspector': {
            'VTKFileInspector': {
                'baseFileName': 'vissteps',
                'dumpDataLinks': 1,
                'dumpReading': 1,
                'dumpReference': 1,
                'dumpIterationInfo': 1
            },
        },
        'outlierFilters': [outlier] if outlier is not None else [],
        'errorMinimizer': 'PointToPointErrorMinimizer',
        'matcher': {
            'KDTreeMatcher': {
                'knn': 1
            }
        },
        'transformationCheckers': [
            {'CounterTransformationChecker': {'maxIterationCount': 40}},
            'DifferentialTransformationChecker'
        ]
    }
    return yaml.dump(config, default_flow_style=False)


def create_models():
    paths, attributes = svg2paths('line.svg')
    model = [[path.start.real, -path.start.imag] for path in paths[0]]

    #     OFFSET = np.array([4, -2])
    #     t = 1 * np.pi / 180 # Theta
    OFFSET = np.array([9, 4])
    t = 1 * np.pi / 180  # Theta
    # Homogeneous transformation matrix
    tf = np.array([[cos(t), -sin(t), OFFSET[0]],
                   [sin(t), cos(t), OFFSET[1]],
                   [0, 0, 1]])

    model = np.array(model)
    # Sample one out of two points
    ref = model[::2, :]
    read = model[1::2, :]

    # Cut point cloud and only keep a small overlaping region
    ref = ref[ref[:, 1] > -88].copy()
    read = read[read[:, 1] < -52].copy()
    # A = model[(model[:, 0] < 73) & (model[:, 1] > -123)] # x < 73 and y > -123
    # B = model[model[:, 1] > -86] # y > -86

    # Apply tf
    read_homo = np.hstack([read, np.ones((read.shape[0], 1))])
    read_homo = read_homo @ tf.transpose()
    read = read_homo[:, :2] / read_homo[:, [-1]]  # Not required
    return ref, read


def create_models_csv():
    ref, read = create_models()
    np.savetxt('ref.csv', ref, delimiter=',')
    np.savetxt('read.csv', read, delimiter=',')
    return ref, read


def do_icp():
    # Write config file
    with open("config.yaml", "w+") as config_file:
        config = generate_config()
        # print(config)
        config_file.write(config)

    delete_vtk_artifacts()

    cmd = "~/local/bin/pmicp --config config.yaml ref.csv read.csv"
    print(cmd)
    try:
        output = subprocess.check_output(cmd, universal_newlines=True, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        print("==config==")
        print(config)
        print("==Error==")
        print(exc.output)
        raise
    print()

    points, segments = parse_vtk('vissteps-link-0.vtk')
    ref, _ = parse_vtk('test_ref.vtk')
    read, _ = parse_vtk('test_data_in.vtk')
    read_last_iter, _ = parse_vtk('test_data_out.vtk')
    return ref, read, segments, read_last_iter


if __name__ == "__main__":

    do_icp()
