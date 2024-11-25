import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm
import random
import os

# generate_polyhedron
def generate_polyhedron_points(num_vertices, total_points):
    # Generate random vertices
    vertices = np.random.uniform(low=-1.0, high=1.0, size=(num_vertices, 3))

    # Create convex hull
    hull = ConvexHull(vertices)

    # Calculate the number of points per face
    num_faces = len(hull.simplices)
    num_points_per_face = total_points // num_faces

    # Generate points on each face
    points = []
    normals = []
    for simplex in hull.simplices:
        face_vertices = vertices[simplex]

        for _ in range(num_points_per_face):
            # Generate random weights
            weights = np.random.uniform(size=3)
            weights /= weights.sum()

            # Generate a random point within the triangle
            point = weights @ face_vertices
            points.append(point)

            # Calculate the normal vector for the point
            normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
            normal /= np.linalg.norm(normal)
            normals.append(normal)

    return np.array(points), np.array(normals)

# generate_ellipsoid
def generate_ellipsoid_points(num_points, radii):
    # Generate random angles
    theta = 2.0 * np.pi * np.random.uniform(size=num_points)  # azimuthal angle
    phi = np.arccos(2.0 * np.random.uniform(size=num_points) - 1.0)  # polar angle

    # Convert spherical coordinates to cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Convert unit sphere points to ellipsoid points
    points = np.vstack((x * radii[0], y * radii[1], z * radii[2])).T

    # Calculate normals
    normals = points / (np.array(radii) ** 2)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    return points, normals

# generate_cylinder
def generate_cylinder_points(num_points, radius, height):
    # Generate random angles
    theta = 2.0 * np.pi * np.random.uniform(size=num_points)  # azimuthal angle

    # Convert polar coordinates to cartesian coordinates
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    z = np.random.uniform(low=-height/2, high=height/2, size=num_points)

    # Combine x, y, z to create points
    points = np.vstack((x, y, z)).T

    # Calculate normals
    normals = points / np.array([radius, radius, 1])
    normals[:, 2] = 0  # z-component of normal is always 0 for a cylinder
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    return points, normals

# generate_cone
def generate_cone_points(num_points, radius, height):
    # Generate random angles
    theta = 2.0 * np.pi * np.random.uniform(size=num_points)  # azimuthal angle

    # Generate random z coordinates
    z = np.random.uniform(low=0, high=height, size=num_points)

    # Convert polar coordinates to cartesian coordinates
    r = (height - z) / height * radius
    x = np.cos(theta) * r
    y = np.sin(theta) * r

    # Combine x, y, z to create points
    points = np.vstack((x, y, z)).T

    # Calculate normals
    normals = points / np.array([np.sqrt(radius**2 + height**2)] * 3)
    normals[:, 2] = height / np.sqrt(radius**2 + height**2)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    return points, normals

# generate_prism
def generate_prism_vertices(num_sides, height):
    vertices = []
    for i in range(num_sides):
        theta = 2 * np.pi * i / num_sides
        x = np.cos(theta)
        y = np.sin(theta)
        vertices.append((x, y, 0))  # Bottom vertex
        vertices.append((x, y, height))  # Top vertex
    return vertices

def generate_prism_faces(num_sides):
    faces = []
    for i in range(num_sides):
        bottom_vertex_index = 2 * i
        top_vertex_index = bottom_vertex_index + 1
        next_bottom_vertex_index = (bottom_vertex_index + 2) % (2 * num_sides)
        next_top_vertex_index = (next_bottom_vertex_index + 1) % (2 * num_sides)
        face1 = [bottom_vertex_index, top_vertex_index, next_top_vertex_index]
        face2 = [bottom_vertex_index, next_top_vertex_index, next_bottom_vertex_index]
        faces.append(face1)
        faces.append(face2)
    return faces

def generate_prism_point_cloud(num_sides, height, num_points):
    vertices = generate_prism_vertices(num_sides, height)
    faces = generate_prism_faces(num_sides)
    point_cloud = []
    normals = []
    for face in faces:
        v0 = np.array(vertices[face[0]])
        v1 = np.array(vertices[face[1]])
        v2 = np.array(vertices[face[2]])
        normal = np.cross(v1 - v0, v2 - v0)  # Calculate the normal vector of the surface
        for _ in range(int(num_points/len(faces))):
            b1 = np.random.uniform(0, 1)
            b2 = np.random.uniform(0, 1)
            if b1 + b2 > 1:
                b1 = 1 - b1
                b2 = 1 - b2
            point = v0 + b1 * (v1 - v0) + b2 * (v2 - v0)
            point_cloud.append(point)
            normals.append(normal)
    return point_cloud, normals

# generate_pyramid
def generate_pyramid_vertices(num_sides, height):
    vertices = []
    for i in range(num_sides):
        theta = 2 * np.pi * i / num_sides
        x = np.cos(theta)
        y = np.sin(theta)
        vertices.append((x, y, 0))  # Bottom vertex
    vertices.append((0, 0, height))  # Top vertex
    return vertices


def generate_pyramid_faces(num_sides):
    faces = []
    for i in range(num_sides):
        face = [i, (i + 1) % num_sides, num_sides]  # Bottom triangle
        faces.append(face)
    bottom_face = list(range(num_sides))  # Bottom polygon
    faces.append(bottom_face)
    return faces


def generate_pyramid_point_cloud(num_sides, height, num_points):
    vertices = generate_pyramid_vertices(num_sides, height)
    faces = generate_pyramid_faces(num_sides)
    point_cloud = []
    normals = []
    for face in faces:
        v0 = np.array(vertices[face[0]])
        v1 = np.array(vertices[face[1]])
        v2 = np.array(vertices[face[2]])
        normal = np.cross(v1 - v0, v2 - v0)  # Calculate the normal vector of the surface
        for _ in range(int(num_points/len(faces))):
            b1 = np.random.uniform(0, 1)
            b2 = np.random.uniform(0, 1)
            if b1 + b2 > 1:
                b1 = 1 - b1
                b2 = 1 - b2
            point = v0 + b1 * (v1 - v0) + b2 * (v2 - v0)
            point_cloud.append(point)
            normals.append(normal)
    return point_cloud, normals


def generate_unique_lists(num_lists, list_lengths, num_range):
    unique_lists = []
    while len(unique_lists) < num_lists:
        list_length = random.choice(list_lengths)
        new_list = sorted(random.choices(range(num_range), k=list_length))
        if new_list not in unique_lists:
            unique_lists.append(new_list)
    return unique_lists

# define rules
rules = {
    'rule1': lambda shape: (shape * np.random.uniform(0.2, 1.0)) + np.random.uniform(-1.0, 1.0, size=3),
    'rule2': lambda shape: np.dot(shape, np.array([[np.cos(np.random.uniform(0, np.pi)), -np.sin(np.random.uniform(0, np.pi)), 0], [np.sin(np.random.uniform(0, np.pi)), np.cos(np.random.uniform(0, np.pi)), 0], [0, 0, 1]])),
    'rule3': lambda shape: (shape * np.random.uniform(1.0, 2.0)) - np.random.uniform(-1.0, 1.0, size=3),
    'rule4': lambda shape: np.dot(shape, np.array([[np.cos(np.random.uniform(0, np.pi)), -np.sin(np.random.uniform(0, np.pi)), 0], [np.sin(np.random.uniform(0, np.pi)), np.cos(np.random.uniform(0, np.pi)), 0], [0, 0, 1]])),
    'rule5': lambda shape: (shape * np.random.uniform(0.5, 1.0)) + np.random.uniform(-1.0, 1.0, size=3),
    'rule6': lambda shape: shape
}

root_dir = '/root/autodl-tmp/DataSet/BasicShape6/'

# load basic shapes
def gen_dataset():
    dataset = {
        'polyhedron': np.loadtxt(os.path.join(root_dir, 'polyhedron', 'polyhedron_' + str(random.randint(1, 2000) ).zfill(4) + '.txt'), delimiter=',')[:, :3],
        'ellipsoid': np.loadtxt(os.path.join(root_dir, 'ellipsoid', 'ellipsoid_' + str(random.randint(1, 2000) ).zfill(4) + '.txt'), delimiter=',')[:, :3],
        'cylinder': np.loadtxt(os.path.join(root_dir, 'cylinder', 'cylinder_' + str(random.randint(1, 2000) ).zfill(4) + '.txt'), delimiter=',')[:, :3],
        'basiccone': np.loadtxt(os.path.join(root_dir, 'basiccone', 'basiccone_' + str(random.randint(1, 2000) ).zfill(4) + '.txt'), delimiter=',')[:, :3],
        'pyramid': np.loadtxt(os.path.join(root_dir, 'pyramid', 'pyramid_' + str(random.randint(1, 2000) ).zfill(4) + '.txt'), delimiter=',')[:, :3],
        'prism': np.loadtxt(os.path.join(root_dir, 'prism', 'prism_' + str(random.randint(1, 2000) ).zfill(4) + '.txt'), delimiter=',')[:, :3],
    }
    return  dataset

# define shape assembly
def combine_shapes(shape_list, rule_list):
    combined_shape = None
    for index, shape in enumerate(shape_list):
        if combined_shape is None:
            combined_shape = rules[rule_list[index]](gen_dataset()[shape])
        else:
            combined_shape = np.vstack([combined_shape, rules[rule_list[index]](gen_dataset()[shape])])
    return combined_shape

from multiprocessing import Pool

# generate shape assemblies
def generate_shapes(shape_index):
    print('Gen Shape' + str(shape_index) + ':')
    if not os.path.exists(os.path.join(root_dir, 'shape' + str(shape_index))):
        os.mkdir(os.path.join(root_dir, 'shape' + str(shape_index)))
    shape_list = ['polyhedron', 'ellipsoid', 'cylinder', 'basiccone', 'pyramid', 'prism']
    rule_list = ['rule1', 'rule2', 'rule3', 'rule4', 'rule5', 'rule6']
    k_items = random.randint(2, 4)
    shape_list = random.choices(shape_list, k=k_items)
    rule_list = random.choices(rule_list, k=k_items)

    for i in tqdm(range(2000)):
        new_shape = combine_shapes(shape_list, rule_list)
        np.savetxt(os.path.join(root_dir, 'shape' + str(shape_index), 'shape' + str(shape_index) + '_' + str(i + 1).zfill(4) + '.txt'), new_shape, delimiter=',', fmt='%f')

if __name__ == '__main__':

# step1: generate basic shapes
    model_name = 'pyramid'
    for i in tqdm(range(2000)):
        # points, normals = generate_polyhedron_points(random.randint(4, 10), 10000)
        # points, normals = generate_ellipsoid_points(10000, [random.uniform(0.05, 1), random.uniform(0.05, 1), random.uniform(0.05, 1)])
        # points, normals = generate_cylinder_points(10000, random.uniform(0.05, 2), random.uniform(0.05, 2))
        # points, normals = generate_cone_points(10000, random.uniform(0.05, 2), random.uniform(0.05, 2))
        # points, normals = generate_pyramid_points(10000, random.uniform(0.05, 2), random.uniform(0.05, 2))
        points, normals = generate_pyramid_point_cloud(random.randint(3, 10), random.uniform(0.05, 5), 10000)
        data = np.hstack((points, normals))
        np.savetxt(os.path.join(root_dir, model_name, model_name + '_' + str(i + 1).zfill(4) + '.txt'), data, delimiter=',', fmt='%f')

# step2: generate shape assemblies
    # with Pool() as p:
    #    p.map(generate_shapes, range(20))


