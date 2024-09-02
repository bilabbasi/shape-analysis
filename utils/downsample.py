import numpy as np
# from scipy.spatial import cKDTree

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
        self.vertices = [Vertex(v) for v in vertices]
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
                    edges[(v1, v2)] = Edge(self.vertices[v1], self.vertices[v2])
        return list(edges.values())

    def compute_vertex_quadrics(self):
        for face in self.faces:
            v1, v2, v3 = [self.vertices[i].position for i in face]
            normal = np.cross(v2 - v1, v3 - v1)
            normal /= np.linalg.norm(normal)
            a, b, c, d = *normal, -np.dot(normal, v1)
            q = np.array([[a*a, a*b, a*c, a*d],
                          [a*b, b*b, b*c, b*d],
                          [a*c, b*c, c*c, c*d],
                          [a*d, b*d, c*d, d*d]])
            for i in face:
                self.vertices[i].q += q

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

        vertex_map = {i: i for i in range(len(self.vertices))}
        removed_vertices = set()

        while len(self.vertices) - len(removed_vertices) > target_vertices:
            edge = self.edges.pop(0)
            v1_idx, v2_idx = [self.vertices.index(v) for v in edge.vertices]

            # Skip if either vertex has been removed
            if v1_idx in removed_vertices or v2_idx in removed_vertices:
                continue

            # Merge v2 into v1
            self.vertices[v1_idx].position = edge.target
            self.vertices[v1_idx].q += self.vertices[v2_idx].q
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
        self.vertices = [self.vertices[i] for i in new_vertex_map]
        self.faces = [[new_vertex_map[vertex_map[i]] for i in face] for face in self.faces]

        return np.array([v.position for v in self.vertices]), self.faces

def downsample_mesh(vertices, faces, target_vertices):
    mesh = Mesh(vertices, faces)
    return mesh.simplify(target_vertices)

def read_off_file(file_path):
    with open(file_path, 'r') as file:
        if file.readline().strip() != 'OFF':
            raise ValueError('The file does not start with OFF')
        
        n_verts, n_faces, n_edges = map(int, file.readline().split())

        vertices = []
        for _ in range(n_verts):
            vertex = list(map(float, file.readline().split()))
            vertices.append(vertex)

        faces = []
        for _ in range(n_faces):
            face = list(map(int, file.readline().split()))
            if face[0] != 3:
                raise ValueError('Only triangular meshes are supported')
            faces.append(face[1:])

    return np.array(vertices), np.array(faces)

def process_and_downsample_off(input_file, target_vertices):
    vertices, faces = read_off_file(input_file)
    new_vertices, new_faces = downsample_mesh(vertices, faces, target_vertices)
    return new_vertices, new_faces

def write_off_file(output_file, vertices, faces):
    with open(output_file, 'w') as file:
        file.write("OFF\n")
        file.write(f"{len(vertices)} {len(faces)} 0\n")
        
        for vertex in vertices:
            file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        for face in faces:
            file.write(f"3 {face[0]} {face[1]} {face[2]}\n")


if __name__ == "__main__":
    # Example usage
    input_file = "data/COSEG/tele_aliens/shapes/10.off"
    output_file = "output_mesh.off"
    target_vertices = 1000

    new_vertices, new_faces = process_and_downsample_off(input_file, target_vertices)
    write_off_file(output_file, new_vertices, new_faces)

    print(f"Original vertices: {len(read_off_file(input_file)[0])}")
    print(f"Original faces: {len(read_off_file(input_file)[1])}")
    print(f"New vertices: {len(new_vertices)}")
    print(f"New faces: {len(new_faces)}")
    print(f"Downsampled mesh written to {output_file}")