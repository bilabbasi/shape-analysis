import matplotlib.pyplot as plt
import numpy as np

cmap = plt.get_cmap("viridis")

from typing import NoReturn, Tuple

def seg2txt(seg: list[str|int], fname: str) -> None:
    lines = [f"{item}\n" for item in seg]
    with open(fname, 'w') as file:
        file.writelines(lines)

def mesh2off(vertices, faces, fname: str="mesh.off"):
    with open(fname, "w") as file:
        file.write("OFF\n")
        file.write(f"{len(vertices)} {len(faces)} 0\n")

        for vertex in vertices:
            file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        for face in faces:
            file.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def read_off(file_path: str) -> Tuple[np.array, np.array]:
    with open(file_path, "r") as file:
        if file.readline().strip() != "OFF":
            raise ValueError("The file does not start with OFF")

        n_verts, n_faces, n_edges = map(int, file.readline().split())

        vertices = []
        for _ in range(n_verts):
            vertex = list(map(float, file.readline().split()))
            vertices.append(vertex)

        faces = []
        for _ in range(n_faces):
            face = list(map(int, file.readline().split()))
            if face[0] != 3:
                raise ValueError("Only triangular meshes are supported")
            faces.append(face[1:])

    return np.array(vertices), np.array(faces)


def mesh2obj(
    vertices: np.array, faces: np.array, fname: str = "mesh.obj", shift: int = 1
) -> NoReturn:
    """
    Returns a .obj with the input vertices and faces.

    :param vertices: nx3 np.array of vertices
    :param faces: mx3 np.array of indices indicates triangulation of ``vertices``
    :param fname: File for storing .obj
    :param shift: Value to shift face indices by
    """
    # Default shift is one because we assume that a reader such as blender will assume faces
    # start their index at 1.
    mesh_obj = open(fname, "w")
    for v in vertices:
        print(f"v {v[0]} {v[1]} {v[2]}", file=mesh_obj)
    for f in faces:
        print(f"f {f[0]+shift} {f[1]+shift} {f[2]+shift}", file=mesh_obj)
    mesh_obj.close()


def pcl2ply(vertices: np.array, fname: str = "pcl.ply") -> NoReturn:
    """
    Returns a .ply with the input vertex point cloud.

    :param vertices: nx3 np.array of vertices
    :param fname: File for storing .ply
    """
    ply = open(fname, "w")
    print("ply", file=ply)
    print("format ascii 1.0", file=ply)
    print(f"element vertex {len(vertices.squeeze())}", file=ply)
    print("property float x", file=ply)
    print("property float y", file=ply)
    print("property float z", file=ply)
    print("end_header", file=ply)
    for v in vertices.squeeze():
        print(f"{v[0]} {v[1]} {v[2]}", file=ply)
    ply.close()


def mesh2ply(
    vertices: np.array,
    faces: np.array,
    weights: np.array,
    weights_to_colours: bool = True,
    fname: str = "mesh.ply",
) -> NoReturn:
    """
    Returns a .ply with the input vertices, faces, and weights (per vertex).

    :param vertices: nx3 np.array of vertices
    :param faces: mx3 array of indices indicating triangulation of ``vertices``
    :param weights: Weights per vertex
    :param weights_to_colours: Boolean indicating to convert the weights to RGB values
    :param fname: File for storing .ply
    """
    ply = open(fname, "w")
    print("ply", file=ply)
    print("format ascii 1.0", file=ply)
    print(f"element vertex {len(vertices)}", file=ply)
    print("property float x", file=ply)
    print("property float y", file=ply)
    print("property float z", file=ply)
    print("property uchar red", file=ply)
    print("property uchar green", file=ply)
    print("property uchar blue", file=ply)
    print(f"element face {len(faces)}", file=ply)
    print("property list uint8 int32 vertex_indices", file=ply)
    print("end_header", file=ply)

    if weights_to_colours:
        colours = (weights - weights.min()) / (weights.max() - weights.min())
        colours = cmap(colours)
        colours *= 255
    else:
        colours = weights

    for v, c in zip(vertices, colours):
        print(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}", file=ply)
    for f in faces:
        print(f"3 {int(f[0])} {int(f[1])} {int(f[2])}", file=ply)
    ply.close()
