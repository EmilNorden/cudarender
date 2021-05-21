#include "device_mesh.cuh"
#include "device_stack.cuh" // For kd tree traversal

using namespace std;

template<typename T>
struct Tuple {
    T item1;
    T item2;
};

struct TreeNode {
    Axis plane_surface_normal;
    glm::vec3 plane_position;
    TreeNode *left;
    TreeNode *right;
    TriangleFace *faces;
    int face_count;
};

struct NodeSearchData {
    TreeNode *node;
    float tmin;
    float tmax;
};

__device__ bool
intersect_node(TreeNode *node, const WorldSpaceRay &ray, float &tmin, float &tmax, DeviceStack<NodeSearchData> &nodes);

__device__ bool
intersect_leaf(TreeNode *node, const WorldSpaceRay &ray, float &tmin, float &tmax, DeviceStack<NodeSearchData> &nodes);

#define EPSILON 9.99999997475243E-07


__device__ bool hit_aabb(const WorldSpaceRay &ray, const AABB &aabb, float &out_near, float &out_far) {
    out_near = FLT_MIN;
    out_far = FLT_MAX;
    for (auto i = 0; i < 3; ++i) {
        /*if glm::abs(self.direction[i]) < EPSILON {
            if self.origin[i] < bounds.min()[i] as f32 || self.origin[i] > bounds.max()[i] as f32 {
                return false
            }
        }
         */
        auto t1 = (aabb.min()[i] - ray.origin().as_vec3()[i]) / ray.direction()[i];// / self.direction[i];
        auto t2 = (aabb.max()[i] - ray.origin().as_vec3()[i]) / ray.direction()[i]; // / self.direction[i];

        if (t1 > t2) {
            auto temp = t1;
            t1 = t2;
            t2 = temp;
        }

        if (t1 > out_near) {
            out_near = t1;
        }

        if (t2 < out_far) {
            out_far = t2;
        }

        if (out_near > out_far || out_far < 0.0f) {
            return false;
        }
    }

    return true;
}

__device__ bool hit_triangle(const WorldSpaceRay &ray, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, float &out_distance) {
    // Find vectors for two edges sharing V1
    glm::vec3 e1 = v2 - v1;
    glm::vec3 e2 = v3 - v1;
    // Begin calculating determinant - also used to calculate u parameter
    glm::vec3 P = glm::cross(ray.direction(), e2); // m_direction.cross(e2);
    // if determinant is near zero, ray lies in plane of triangle

    float det = glm::dot(e1, P); // e1.dot(P);

    /*if (det > -EPSILON && det < EPSILON)
        return false;*/

    // BACK-FACE CULLING

    /*if (inside_geometry && det > EPSILON) {
        return false;
    }*/
    /*if (!inside_geometry && det < EPSILON) {
        return false;
    }*/

    float inv_det = 1.0f / det;

    // calculate distance from V1 to ray origin
    glm::vec3 T = ray.origin().as_vec3() - v1;

    // Calculate u parameter and test bound
    float u = glm::dot(T, P) * inv_det;
    // The intersection lies outside of the triangle
    if (u < 0.f || u > 1.f)
        return false;

    // Prepare to test v parameter
    glm::vec3 Q = glm::cross(T, e1); // T.cross(e1);

    // Calculate V parameter and test bound
    float v = glm::dot(ray.direction(), Q) * inv_det;
    // The intersection lies outside of the triangle
    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = glm::dot(e2, Q) * inv_det;

    if (t > EPSILON) { // ray intersection
        out_distance = t;
        // *dist = t;
        // *result_u = u;
        // *result_v = v;
        return true;
    }

    // No hit, no win
    return false;
}

__host__ IndexedDeviceMesh::IndexedDeviceMesh(const std::vector<glm::vec3> &vertices,
                                              const std::vector<TriangleFace> &faces)
        : m_bounds(vertices) {

    // cudaMalloc(&m_vertices, sizeof(glm::vec3) * vertices.size());
    cudaMallocManaged(&m_vertices, sizeof(glm::vec3) * vertices.size());
    cudaMemcpy(m_vertices, vertices.data(), sizeof(glm::vec3) * vertices.size(), cudaMemcpyHostToHost);
    m_vertex_count = vertices.size();

    // cudaMalloc(&m_faces, sizeof(TriangleFace) * faces.size());

    // We shouldnt need this anymore, this goes into the kdtree
    cudaMallocManaged(&m_faces, sizeof(TriangleFace) * faces.size());
    cudaMemcpy(m_faces, faces.data(), sizeof(TriangleFace) * faces.size(), cudaMemcpyHostToHost);
    m_face_count = faces.size();

    cudaMallocManaged(&m_root, sizeof(TreeNode));
    auto faces_copy = faces;
    build_node(*m_root, faces_copy, Axis::X);
}

void IndexedDeviceMesh::build_node(TreeNode &node, std::vector<TriangleFace> &faces, Axis current_axis) {

    if (faces.size() < 100) {
        cudaMalloc(&node.faces, sizeof(TriangleFace) * faces.size());
        cudaMemcpy(node.faces, faces.data(), sizeof(TriangleFace) * faces.size(), cudaMemcpyHostToDevice);
        node.face_count = faces.size();
        node.left = nullptr;
        node.right = nullptr;
        return;
    }

    static glm::vec3 axis_normals[3] = {
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f)
    };

    std::sort(faces.begin(), faces.end(), [&](const TriangleFace &a, const TriangleFace &b) {
        // Just use the first vertex for each face.
        // This is only used to find a suitable median position for the splitting plane.
        // It determines how balanced our tree will be.

        auto a_v0 = m_vertices[a.i0];
        auto b_v0 = m_vertices[b.i0];

        // TODO: Replace this with a simple multiplication since we are working with axis aligned vectors.
        // TODO: IE the projection of {1.03; 8.00; -34.01} onto {1.0; 0.0, 0.0} is simply the value of X, 1.03.
        auto a_projection = glm::dot(a_v0, axis_normals[static_cast<int>(current_axis)]);
        auto b_projection = glm::dot(b_v0, axis_normals[static_cast<int>(current_axis)]);

        return a_projection < b_projection;
    });

    auto half_size = faces.size() / 2;
    auto median_point = m_vertices[faces[half_size].i0];
    auto median_projection = glm::dot(median_point, axis_normals[static_cast<int>(current_axis)]);

    // If there was no chance of triangles belonging on both sides (intersecting the splitting plane)
    // we could just do a naive split like below. Instead we have to loop through and possibly duplicate
    // std::vector<TriangleFace> left_side(faces.begin(), faces.begin() + half_size);
    // std::vector<TriangleFace> right_side(faces.begin() + half_size, faces.end());

    std::vector<TriangleFace> left_side;
    std::vector<TriangleFace> right_side;

    left_side.reserve(half_size);
    right_side.reserve(half_size);

    for (auto &face : faces) {
        auto v0_projection = glm::dot(m_vertices[face.i0], axis_normals[static_cast<int>(current_axis)]);
        auto v1_projection = glm::dot(m_vertices[face.i1], axis_normals[static_cast<int>(current_axis)]);
        auto v2_projection = glm::dot(m_vertices[face.i2], axis_normals[static_cast<int>(current_axis)]);

        if (v0_projection >= median_projection ||
            v1_projection >= median_projection ||
            v2_projection >= median_projection) {
            right_side.push_back(face);
        }

        if (v0_projection < median_projection ||
            v1_projection < median_projection ||
            v2_projection < median_projection) {
            left_side.push_back(face);
        }
    }

    node.plane_surface_normal = current_axis;
    node.plane_position = median_point; // TODO: This could really just be a single float
    node.faces = nullptr;
    node.face_count = 0;
    cudaMallocManaged(&node.left, sizeof(TreeNode));
    cudaMallocManaged(&node.right, sizeof(TreeNode));
    build_node(*node.left, left_side, static_cast<Axis>((current_axis + 1) % 3));
    build_node(*node.right, right_side, static_cast<Axis>((current_axis + 1) % 3));
}

__device__ bool IndexedDeviceMesh::intersect(const WorldSpaceRay &ray, float &out_distance) {
    out_distance = FLT_MAX;
    float tmin = 0.0f;
    float tmax = 0.0f;
    if (!hit_aabb(ray, m_bounds, tmin, tmax)) {
        return false;
    }

    DeviceStack<NodeSearchData> nodes;
    return intersect_node(m_root, ray, tmin, tmax, nodes);
}

__device__ Tuple<TreeNode *> order_subnodes(const WorldSpaceRay &ray, TreeNode *node, float tmin, float tmax) {
    auto axis = static_cast<int>(node->plane_surface_normal);
    auto tmin_axis = ray.origin().as_vec3()[axis] + (ray.direction()[axis] * tmin);
    auto tmax_axis = ray.origin().as_vec3()[axis] + (ray.direction()[axis] * tmax);

    auto split_value = node->plane_position[axis];

    if (tmin_axis < split_value && tmax_axis < split_value) {
        return {
                node->left,
                node->right
        };
    } else if (tmin_axis >= split_value && tmax_axis >= split_value) {
        return {
                node->right,
                node->left
        };
    } else {
        return {
                node->left,
                node->right
        };
    }


}

__device__ bool
intersect_split(TreeNode *node, const WorldSpaceRay &ray, float &tmin, float &tmax, DeviceStack<NodeSearchData> &nodes) {
    auto a = (int) node->plane_surface_normal;
    auto thit = (node->plane_position[a] - ray.origin().as_vec3()[a]) / ray.direction()[a];
    auto nodes_to_search = order_subnodes(ray, node, tmin, tmax);

    if (thit >= tmax || thit < 0) {
        return intersect_node(nodes_to_search.item1, ray, tmin, tmax, nodes);
    }
    else if(thit <= tmin) {
        return intersect_node(nodes_to_search.item2, ray, tmin, tmax, nodes);
    }
    else {
        nodes.push({
            nodes_to_search.item2,
            thit,
            tmax
        });
        return intersect_node(nodes_to_search.item1, ray, tmin, thit, nodes);
    }
}

__device__ bool
intersect_node(TreeNode *node, const WorldSpaceRay &ray, float &tmin, float &tmax, DeviceStack<NodeSearchData> &nodes) {
    if (node->faces) {
        return intersect_leaf(node, ray, tmin, tmax, nodes);
    } else {
        return intersect_split(node, ray, tmin, tmax, nodes);
    }
}

__device__ bool
intersect_leaf(TreeNode *node, const glm::vec3 *vertices, const WorldSpaceRay &ray, float &tmin, float &tmax,
               DeviceStack<NodeSearchData> &nodes) {
    auto success = false;
    for (auto i = 0; i < node->face_count; ++i) {
        auto v0 = vertices[node->faces[i].i0];
        auto v1 = vertices[node->faces[i].i1];
        auto v2 = vertices[node->faces[i].i2];

        float hit_distance = 0.0f;

        auto hit_result = hit_triangle(ray, v0, v1, v2, hit_distance);

        if (hit_result && hit_distance < tmax) {
            tmax = hit_distance;
            success = true;
        }
    }

    if (!success) {
        if (nodes.is_empty()) {
            return false;
        } else {
            auto node = nodes.pop();
            return intersect_node(node.node, ray, node.tmin, node.tmax, nodes);
        }
    }

    return success;
}