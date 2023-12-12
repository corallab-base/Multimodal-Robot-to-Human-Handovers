# Imports
import sys
import fcl
import math
import numpy as np
import pyvista as pv
from stl import mesh
from kinematics import fk
from constants import plane, robot_offset
from scipy.spatial.transform import Rotation as R

# Importing the OMPL collision management system
sys.path.append('/home/corallab/Downloads/ompl/py-bindings')
import ompl.base as ob
import ompl.geometric as og




# Creating a class to read stl files into a mesh
class STL_Reader:
    # constructor
    def __init__(self, file_name):
        self.vertices_ = None
        self.faces_ = None

        dicts = {}
        reverse_dicts = {}

        mesh_info = mesh.Mesh.from_file(file_name)
        face_data = mesh_info.vectors
        num_face, _, _, = face_data.shape

        # finding/storing faces and vertices
        index = 0
        for i in range(num_face):
            for j in range(3):
                vertex = tuple(face_data[i][j])
                if vertex not in dicts:
                    dicts[vertex] = index
                    reverse_dicts[index] = vertex
                    index += 1

        # getting vertices of the mesh
        vertices = []
        for i in range(len(reverse_dicts)):
            vertices.append(reverse_dicts[i])
        self.vertices_ = np.array(vertices)

        # getting the faces of the mesh
        faces = []
        for i in range(num_face):
            face_index = []
            for j in range(3):
                face_index.append(dicts[tuple(face_data[i][j])])
            faces.append(face_index)
        self.faces_ = np.array(faces)


    # translation should be defined in cartesian space (x, y, z) dim(1,3)
    def transform(self, rotation, translation):
        if self.vertices_.any() and self.faces_.any():
            translation = np.array(translation)
            for i in range(len(self.vertices_)):
                self.vertices_[i] = rotation.apply(self.vertices_[i])
                self.vertices_[i] += translation


    # get vertices defined by the mesh
    def get_vertices(self):
        return self.vertices_


    # get faces defined by the mesh
    def get_faces(self):
        return self.faces_


    # write out the mesh using the vertices and faces info extracted
    def write_to_file(self, file_name):
        if self.vertices_.any() and self.faces_.any():
            omesh = mesh.Mesh(np.zeros(self.faces_.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(self.faces_):
                for j in range(3):
                    omesh.vectors[i][j] = self.vertices_[f[j],:]

            omesh.save(file_name)
        else:
            print("Not enough data to define a mesh!")
            sys.exit(1)


    # Creates an fcl model from vertices and faces
    def get_fcl_collision_object(self, x, y, z, rw, rx, ry, rz):
        model = fcl.BVHModel()
        model.beginModel(len(self.vertices_), len(self.faces_))
        model.addSubModel(self.vertices_, self.faces_)
        model.endModel()

        translation = np.array([x, y, z])
        rotation = R.from_quat([rw, rx, ry, rz]).as_matrix()
        tf = fcl.Transform(rotation, translation)
        return fcl.CollisionObject(model, tf)




# Stores robot configuration & assosiated transformations for collision detection
class robot_arm_configuration:
    def __init__(self):        
        # Setting up all the collision meshes
        asset_path = "/home/corallab/guna/riemannian-motion-control/assets/"
        ur5e_collision_parts = ["meshes/collision/base.stl",
                                "meshes/collision/shoulder.stl",
                                "meshes/collision/upperarm.stl",
                                "meshes/collision/forearm.stl",
                                "meshes/collision/wrist1.stl",
                                "meshes/collision/wrist2.stl",
                                "meshes/collision/wrist3.stl",
                                "meshes/2f85/robotiq_85_base_link_coarse.STL",
                                "meshes/2f85/inner_knuckle_coarse.STL",
                                "meshes/2f85/inner_finger_coarse.STL",
                                "meshes/2f85/outer_knuckle_coarse.STL",
                                "meshes/2f85/inner_knuckle_coarse.STL",
                                "meshes/2f85/inner_finger_coarse.STL",
                                "meshes/2f85/outer_knuckle_coarse.STL",
                                "meshes/2f85/outer_finger_coarse.STL",
                                "meshes/2f85/outer_finger_coarse.STL"]
        
        self.ur5e_rotations = [R.from_euler('x',  [90], degrees = True),
                        R.from_euler('xy', [90, 180], degrees = True),
                        R.from_euler('xy', [180, 180], degrees = True),
                        R.from_euler('z',  [-180], degrees = True),
                        R.from_euler('x',  [-180], degrees = True),
                        R.from_euler('x',  [90], degrees = True),
                        R.from_euler('z',  [-90], degrees = True),
                        R.from_euler('xyz', [0, 0, 0], degrees = True),
                        R.from_euler('xyz', [0, 0, 0], degrees = True),
                        R.from_euler('xyz', [0, 0, 0], degrees = True),
                        R.from_euler('xyz', [0, 0, 0], degrees = True),
                        R.from_euler('xyz', [0, 0, 0], degrees = True),
                        R.from_euler('xyz', [0, 0, 0], degrees = True),
                        R.from_euler('xyz', [0, 0, 0], degrees = True),
                        R.from_euler('xyz', [0, 0, 0], degrees = True),
                        R.from_euler('xyz', [0, 0, 0], degrees = True)]
                
        self.ur5e_translations = [robot_offset,
                            [0, 0, 0],
                            [0, -0.138, 0],
                            [0, -0.007, 0],
                            [0, 0.127, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]
        
        # Generating fcl models
        self.ur5e_collision_models = []
        for i in range(len(ur5e_collision_parts)):
            parts_path = ur5e_collision_parts[i]
            collision_mesh = STL_Reader(asset_path + parts_path)
            model = fcl.BVHModel()
            collision_mesh.transform(self.ur5e_rotations[i], self.ur5e_translations[i])
            verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
            model.beginModel(len(verts), len(tris))
            model.addSubModel(verts, tris)
            model.endModel()
            self.ur5e_collision_models.append(model)


    # Checks if the current state has no collisions
    def has_collisions(self, dof_state, env_objects):
        # Necessary variables
        ur5e_self_col = []
        pose_array = fk(dof_state)

        # Updating models' transformations 
        for t in range(16):
            rotation = np.array(pose_array[t][1])
            translation = np.array(pose_array[t][0])
            
            r1 = R.from_quat(rotation)
            tf = fcl.Transform(r1.as_matrix(), translation)

            ur5e_self_col.append(fcl.CollisionObject(self.ur5e_collision_models[t], tf))

        # Setting up to check for collisions
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()

        for t in range(7):
            # Checking for ground collisions for all parts other than base
            if t != 0:
                if fcl.collide(ur5e_self_col[t], plane, request, result):
                    return True
            
            # Checking for self collisions
            for q in range(t + 2, 7):
                if fcl.collide(ur5e_self_col[t], ur5e_self_col[q], request, result):
                    return True

        # Checking for collisions with environment objects
        manager1 = fcl.DynamicAABBTreeCollisionManager()
        manager1.registerObjects(ur5e_self_col)
        manager1.setup()

        manager2 = fcl.DynamicAABBTreeCollisionManager()
        manager2.registerObjects(env_objects)
        manager2.setup()

        req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)
        rdata = fcl.CollisionData(request = req)
        manager1.collide(manager2, rdata, fcl.defaultCollisionCallback)
        return rdata.result.is_collision
    

    def visualize_arm(self, angles):
        # Path to assets
        file_path = "/home/corallab/guna/riemannian-motion-control/assets/meshes/all/"
        
        # All relevant parts
        link_names = ['base', 'shoulder', 'upperarm', 'forearm', 'wrist1', 'wrist2', 'wrist3']
        gripper_parts = ['robotiq_85_base_link', 'inner_knuckle', 'inner_finger', 'outer_knuckle', 'outer_finger']

        link_points, collision_models, gripper_points = {}, {}, set()
        gripper_translation =   [[0, 0, 0], 
                                [0.013, 0, 0.069],
                                [0.047, 0, 0.115],
                                [0.030, 0, 0.063],
                                [0.062, 0, 0.061],
                                [-0.013, 0, 0.069],
                                [-0.047, 0, 0.115],
                                [-0.031, 0, 0.063],
                                [-0.062, 0, 0.061]]
        gripper_rotation = [[0, 0, 0, 1],
                            [0, 0, 0, 1],
                            [0, 0, 0, 1],
                            [0, 0, 0, 1],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 1, 0],
                            [0, 0, 1, 0],
                            [0, 0, 1, 0]]
    

        
        index_addon = np.array([0, 0, 0])
        gripper_vertices = np.array([]).reshape(0, 3)
        gripper_faces = np.array([]).reshape(0, 3)
        for t in range(len(gripper_parts)):
            part_mesh = STL_Reader(file_path + gripper_parts[t] + '_coarse.STL')
            mesh = pv.read(file_path + gripper_parts[t] + '_coarse.STL')

            min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
            max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
            for tx, ty, tz in part_mesh.get_vertices():
                min_x = min(min_x, tx)
                min_y = min(min_y, ty)
                min_z = min(min_z, tz)
                max_x = max(max_x, tx)
                max_y = max(max_y, ty)
                max_z = max(max_z, tz)

            bounding_points = []
            for tx in range(math.floor(min_x / 0.01), math.ceil(max_x / 0.01) + 1):
                for ty in range(math.floor(min_y / 0.01), math.ceil(max_y / 0.01) + 1):
                    for tz in range(math.floor(min_z / 0.01), math.ceil(max_z / 0.01) + 1):
                        bounding_points.append([tx * 0.01, ty * 0.01, tz * 0.01])

            bounding_points_poly = pv.PolyData(bounding_points)
    
            select = bounding_points_poly.select_enclosed_points(mesh)
            selected_points = select['SelectedPoints']
            temp_link_point_set = set()

            local_translation_1, local_rotation_1 = None, None
            local_translation_2, local_rotation_2 = None, None

            if t != 0:

                local_translation_1 = np.array(gripper_translation[t])
                local_translation_2 = np.array(gripper_translation[t + 4])
                local_rotation_1 = R.from_quat(gripper_rotation[t])
                local_rotation_2 = R.from_quat(gripper_rotation[t + 4])

                part_mesh2 = STL_Reader(file_path + gripper_parts[t] + '_coarse.STL')
                part_mesh.transform(local_rotation_1, local_translation_1)
                part_mesh2.transform(local_rotation_2, local_translation_2)

                #left side
                left_part_vertices = part_mesh.get_vertices()
                left_part_faces = part_mesh.get_faces()
                left_part_faces += index_addon
                vertex_count, _ = left_part_vertices.shape
                    
                gripper_vertices = np.concatenate((gripper_vertices, left_part_vertices), axis = 0)
                gripper_faces = np.concatenate((gripper_faces, left_part_faces), axis = 0)

                index_addon += vertex_count

                #right side
                right_part_vertices = part_mesh2.get_vertices()
                right_part_faces = part_mesh2.get_faces()
                right_part_faces += index_addon
                vertex_count, _ = right_part_vertices.shape
                gripper_vertices = np.concatenate((gripper_vertices, right_part_vertices), axis = 0)
                gripper_faces = np.concatenate((gripper_faces, right_part_faces), axis = 0)

                index_addon += vertex_count
            else:
                part_vertices = part_mesh.get_vertices()
                part_faces = part_mesh.get_faces()
                part_faces += index_addon
                vertex_count, _ = part_vertices.shape

                gripper_vertices = np.concatenate((gripper_vertices, part_vertices), axis = 0)
                gripper_faces = np.concatenate((gripper_faces, part_faces), axis = 0)

                index_addon += vertex_count
                
            
            for i in range(len(bounding_points)):
                if selected_points[i]:
                    if t == 0:
                        gripper_points.add(tuple(bounding_points[i]))
                    else:
                        left_copy, right_copy = bounding_points[i], bounding_points[i]
                        left_copy = local_rotation_1.apply(left_copy)
                        left_copy += local_translation_1
                        right_copy = local_rotation_2.apply(right_copy)
                        right_copy += local_translation_2
                        gripper_points.add(tuple(left_copy.tolist()))
                        gripper_points.add(tuple(right_copy.tolist()))

        for t in range(len(link_names)):
            link = link_names[t]
            link_mesh = STL_Reader(file_path + link + '.stl')

            min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
            max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
            for tx, ty, tz in link_mesh.get_vertices():
                min_x = min(min_x, tx)
                min_y = min(min_y, ty)
                min_z = min(min_z, tz)
                max_x = max(max_x, tx)
                max_y = max(max_y, ty)
                max_z = max(max_z, tz)

            bounding_points = []
            for tx in range(math.floor(min_x / 0.01), math.ceil(max_x / 0.01) + 1):
                for ty in range(math.floor(min_y / 0.01), math.ceil(max_y / 0.01) + 1):
                    for tz in range(math.floor(min_z / 0.01), math.ceil(max_z / 0.01) + 1):
                        bounding_points.append([tx * 0.01, ty * 0.01, tz * 0.01])

            bounding_points_poly = pv.PolyData(bounding_points)
    
            mesh = pv.read(file_path + link + '.stl')
            select = bounding_points_poly.select_enclosed_points(mesh)
            selected_points = select['SelectedPoints']
            temp_link_point_set = set()

            for i in range(len(bounding_points)):
                if selected_points[i]:
                    temp_points = np.array(bounding_points[i])
                    temp_points = self.ur5e_rotations[t].apply(temp_points)
                    temp_points += self.ur5e_translations[t]
                    temp_points = temp_points.tolist()
                    temp_link_point_set.add(tuple(temp_points))
                    
            link_points[link] = temp_link_point_set

            temp_rotation = self.ur5e_rotations[t]
            temp_translation = self.ur5e_translations[t]
            
            link_mesh.transform(temp_rotation, temp_translation)

            vertices, faces = link_mesh.get_vertices(), link_mesh.get_faces()
            collision_models[link] = [vertices, faces.astype(int)]


        link_points['gripper'] = gripper_points
        link_names.append('gripper')
        collision_models['gripper'] = [gripper_vertices, gripper_faces.astype(int)]
        fcl_models = []
        for link in link_names:
            m = fcl.BVHModel()
            vertices, faces = collision_models[link]
            m.beginModel(len(vertices), len(faces))
            m.addSubModel(vertices, faces)
            m.endModel()
            fcl_models.append(m)
        
        transform_data = fk(angles)[0:6]
        translation = [x[0] for x in transform_data]
        rotation = [x[1] for x in transform_data]
        plotter = pv.Plotter()

        # Constructing the robot mesh
        for i in range(len(rotation)):
            link_name = link_names[i]
            temp_translation = translation[i]
            temp_rotation = R.from_quat(rotation[i])
            temp_vertices, temp_faces = collision_models[link_name]
            face_counts, _ =  temp_faces.shape


            new_vertices = temp_rotation.apply(temp_vertices) + temp_translation
            plot_faces = np.concatenate((np.array([3]*face_counts).reshape(face_counts, 1), temp_faces), axis = 1).astype(int)
            temp_mesh = pv.PolyData(np.array(new_vertices), np.array(plot_faces))
            plotter.add_mesh(temp_mesh, color = '#FF6961')
        
        # Visualizing the robot
        _ = plotter.add_axes(line_width = 5)
        plotter.camera_position = 'yz'
        plotter.set_background('white')
        plotter.show()


# Creating a global robot arm configuration for memory efficiency
rac = robot_arm_configuration()