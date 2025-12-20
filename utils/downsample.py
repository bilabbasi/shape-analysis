import numpy as np
import torch
from utils.mesh import read_off, mesh2ply, mesh2off

class Vertex:
    def __init__(self, position):
        self.position = np.array(position)
        self.q = np.zeros((4, 4))

class Edge:
    def __init__(self, v1, v2):
        self.vertices = (v1, v2)
        self.cost = None
        self.target = None

class Mesh:
    def __init__(self, vertices, faces):
        self._vertices = [Vertex(v) for v in vertices]
        self.faces = faces
        self.edges = self.build_edges()

    def build_edges(self):
        edges = {}
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                if v1 > v2:
                    v1, v2 = v2, v1
                if (v1, v2) not in edges:
                    edges[(v1, v2)] = Edge(self._vertices[int(v1)], self._vertices[int(v2)])
        return list(edges.values())

    def compute_vertex_quadrics(self):
        for face in self.faces:
            v1, v2, v3 = [self._vertices[i].position for i in face]
            normal = np.cross(v2 - v1, v3 - v1)
            normal /= np.linalg.norm(normal)
            a, b, c, d = *normal, -np.dot(normal, v1)
            q = np.array([[a*a, a*b, a*c, a*d],
                          [a*b, b*b, b*c, b*d],
                          [a*c, b*c, c*c, c*d],
                          [a*d, b*d, c*d, d*d]])
            for i in face:
                self._vertices[i].q += q

    def compute_edge_costs(self):
        for edge in self.edges:
            v1, v2 = edge.vertices
            q = v1.q + v2.q
            try:
                target = np.linalg.solve(q[:3, :3], -q[:3, 3])
            except np.linalg.LinAlgError:
                target = (v1.position + v2.position) / 2
            cost = target.dot(q[:3, :3]).dot(target) + 2 * q[:3, 3].dot(target) + q[3, 3]
            edge.cost = cost
            edge.target = target

    def simplify(self, target_vertices):
        self.compute_vertex_quadrics()
        self.compute_edge_costs()
        self.edges.sort(key=lambda e: e.cost)

        vertex_map = {i: i for i in range(len(self._vertices))}
        removed_vertices = set()

        while len(self._vertices) - len(removed_vertices) > target_vertices:
            try:
                edge = self.edges.pop(0)
            except IndexError:
                break
            v1_idx, v2_idx = [self._vertices.index(v) for v in edge.vertices]

            # Skip if either vertex has been removed
            if v1_idx in removed_vertices or v2_idx in removed_vertices:
                continue

            # Merge v2 into v1
            v1_pos = self._vertices[v1_idx].position
            v2_pos = self._vertices[v2_idx].position

            # self.vertices[v1_idx].position = edge.target
            self._vertices[v1_idx].position = (v1_pos + v2_pos)/2
            self._vertices[v1_idx].q += self._vertices[v2_idx].q
            removed_vertices.add(v2_idx)

            # Update vertex map
            for i, v in vertex_map.items():
                if v == v2_idx:
                    vertex_map[i] = v1_idx

            # Update faces
            for face in self.faces:
                face[:] = [vertex_map[i] for i in face]

            # Remove degenerate faces
            self.faces = [face for face in self.faces if len(set(face)) == 3]

        # Remove deleted vertices and update faces
        new_vertex_map = {v: i for i, v in enumerate(set(vertex_map.values()) - removed_vertices)}
        self._vertices = [self._vertices[i] for i in new_vertex_map]
        self.faces = [[new_vertex_map[vertex_map[i]] for i in face] for face in self.faces]

        return np.array([v.position for v in self._vertices]), self.faces, new_vertex_map

    @property
    def vertices(self):
        return torch.tensor([v.position for v in self._vertices], dtype=torch.float)

    @property
    def edge_index(self):
        edge_indices = []
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i].item(), face[(i + 1) % 3].item()
                if v1 > v2:
                    v1, v2 = v2, v1
                edge_indices.append((v1, v2))
        return torch.tensor(list(set(edge_indices)), dtype=torch.long).t().contiguous()

def downsample_mesh(vertices, faces, target_vertices):
    mesh = Mesh(vertices, faces)
    return mesh.simplify(target_vertices)

def process_and_downsample_off(input_file, target_vertices):
    vertices, faces = read_off(input_file)
    new_vertices, new_faces, vertex_map = downsample_mesh(vertices, faces, target_vertices)
    return new_vertices, new_faces, vertex_map

def process_segmentation_map(segmentation_file: str, vertex_map: dict=None):
    with open(segmentation_file, 'r') as f:
        old_segmentation = [int(l.strip()) for l in f.readlines()]

    new_segmentation = None
    if vertex_map is not None:
        new_segmentation = list(range(len(vertex_map)))
        for original_idx, label in enumerate(old_segmentation):
            if original_idx in vertex_map.keys():
                new_segmentation[vertex_map[original_idx]] = label
    return old_segmentation, new_segmentation

if __name__ == "__main__":
    # Example usage
    input_file = "data/COSEG/train/tele_aliens/shapes/1.off"
    gt_file = "data.COSEG/train/tele_aliens/vert_gt/1.seg"

    output_file = "output_mesh.off"
    output_gt = "output_gt.seg"
    target_vertices = 1000

    vertices, faces = read_off(input_file)

    new_vertices, new_faces, vertex_map = process_and_downsample_off(input_file, target_vertices)
    gt, new_gt = process_segmentation_map(gt_file, vertex_map)

    mesh2off(new_vertices, new_faces, output_file)

    mesh2ply(vertices, faces, np.array(gt), fname="output_segmentation_source.ply")
    mesh2ply(new_vertices, new_faces, np.array(new_gt), fname=f"output_segmentation_{target_vertices}.ply")

    print(f"Original vertices: {len(read_off(input_file)[0])}")
    print(f"Original faces: {len(read_off(input_file)[1])}")
    print(f"New vertices: {len(new_vertices)}")
    print(f"New faces: {len(new_faces)}")
    print(f"Downsampled mesh written to {output_file}")
