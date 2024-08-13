import trimesh
import point_cloud_utils as pcu
import numpy as np
import os
import math

def convert():
    import bpy
    # Specify the path to the input GLB file and the output OBJ file
    input_file = "905ea9d5e6be4cc39b4885bb72db0836.glb"
    output_file = "modified_object.stl"

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import the GLB file
    bpy.ops.import_scene.gltf(filepath=input_file)

    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')

    # Select only mesh objects and make one active
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0] if mesh_objects else None

    # Ensure there are selected objects to join
    if len(mesh_objects) > 1:
        bpy.ops.object.join()

    # Export the single object to an OBJ file
    bpy.ops.export_mesh.stl(
        filepath=output_file,
        check_existing=True,
        use_selection=True,  # Ensure only the selected (joined) object is exported
        global_scale=1.0,
        use_scene_unit=False,
        ascii=False,
        use_mesh_modifiers=True,
        axis_forward='Y',
        axis_up='Z'
    )

def normalize(vector):
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0))
def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """
    f = open(path, "w")
    for point in pointcloud:
        f.write(f"v {point[0]} {point[1]} {point[2]}\n")
    f.close()
def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """
    areas = []
    total_area = 0
    for face in faces:
        a = vertices[face[0]]
        b = vertices[face[1]]
        c = vertices[face[2]]
        ab = b-a
        ac = c-a
        angle = angle_between(ab, ac)
        area = np.linalg.norm(ab) * np.linalg.norm(ac) * np.sin(angle) * 0.5
        if math.isnan(area):
            area = 0
        areas.append(area)
        total_area += area
    
    if total_area == 0:
        total_area = 1
        breakpoint()
    
    for i in range(0, len(areas)):
        face = faces[i]
        areas[i] = areas[i] / total_area
    try:
        dist = np.random.choice(len(faces), n_points, p=areas)
    except:
        breakpoint()
    points = []
    rands = np.random.rand(n_points * 2)
    for i in range(0, n_points):
        r = [rands[2*i], rands[2*i+1]]
        root = np.sqrt(r)
        u = 1 - root[0]
        v = root[0] * (1-r[1])
        w = root[0] * r[1]
        face = faces[dist[i]]
        a = vertices[face[0]]
        b = vertices[face[1]]
        c = vertices[face[2]]
        points.append(u*a + v * b + w * c)
    return np.array(points)
# Save a point cloud at the location by sampling points from the mesh
def save(location, name, cat, points = 2048):
    cat = "pointclouds/" + cat

    scene = trimesh.load(location)
    #v, f, n = pcu.load_mesh_vfn(location)
    try:
        #print(scene)
        geometries = list(scene.geometry.values())
        #print(geometries)
        vertice_list = [mesh.vertices for mesh in geometries]
        faces_list = [mesh.faces for mesh in geometries]
        faces_offset = np.cumsum([v.shape[0] for v in vertice_list])
        faces_offset = np.insert(faces_offset, 0, 0)[:-1]
        vertices = np.vstack(vertice_list)
        faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])
        merged__meshes = trimesh.Trimesh(vertices, faces)
        #print(name)
        #print(merged__meshes)
        if True:
            bc = np.array(sample_point_cloud(merged__meshes.vertices,merged__meshes.faces, points))
            #_, bc = pcu.sample_mesh_random(merged__meshes.vertices,merged__meshes.faces, points)
        else:
            x = 1
            #fid, bc = pcu.sample_mesh_random(v, f, 2048)
            #pc = pcu.interpolate_barycentric_coords(f, fid, bc, v)
        np.save(cat + "/" + name, bc)
        #export_pointcloud_to_obj(cat + "/" + name + ".obj", pc)
        return True
    except AttributeError:
        print("Attribute not found, skipping element")
        return False
