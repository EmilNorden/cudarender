#include "device_mesh.cuh"
#include "device_stack.cuh" // For kd tree traversal

using namespace std;

#define STACK_SIZE      20

template<typename T>
struct Tuple {
    T item1;
    T item2;
};

struct TreeNode {
    Axis splitting_axis;
    float splitting_value;
    TreeNode *left;
    TreeNode *right;
    TriangleFace *faces;
    int face_count;
};

struct NodeSearchData {
    TreeNode *node;
    float tmin;
    float tmax;

    __device__ NodeSearchData()
            : node(nullptr), tmin(0), tmax(0) {

    }

    __device__ NodeSearchData(TreeNode *n, float min, float max)
            : node(n), tmin(min), tmax(max) {

    }
};

__device__ bool
intersect_node(TreeNode *node, const WorldSpaceRay &ray, float &tmin, float &tmax,
               DeviceStack<STACK_SIZE, NodeSearchData> &nodes);

__device__ bool
intersect_leaf(TreeNode *node, const WorldSpaceRay &ray, float &tmin, float &tmax,
               DeviceStack<STACK_SIZE, NodeSearchData> &nodes);

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

bool is_sorted(glm::vec3 *verts, std::vector<TriangleFace> &faces) {
    for (auto i = 0; i < faces.size() - 1; ++i) {
        auto current_face = faces[i];
        auto next_face = faces[i + 1];

        if (verts[current_face.i0].x > verts[next_face.i0].x) {
            return false;
        }
    }

    return true;
}

void IndexedDeviceMesh::build_node(TreeNode &node, std::vector<TriangleFace> &faces, Axis current_axis) {

    if (faces.size() < 2000) {
        cudaMalloc(&node.faces, sizeof(TriangleFace) * faces.size());
        cudaMemcpy(node.faces, faces.data(), sizeof(TriangleFace) * faces.size(), cudaMemcpyHostToDevice);
        node.face_count = faces.size();
        node.left = nullptr;
        node.right = nullptr;
        return;
    }

    auto axis = static_cast<int>(current_axis);

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

        return a_v0[axis] < b_v0[axis];
    });

    auto half_size = faces.size() / 2;
    auto median_point = m_vertices[faces[half_size].i0];
    auto splitting_value = median_point[axis];

    // If there was no chance of triangles belonging on both sides (intersecting the splitting plane)
    // we could just do a naive split like below. Instead we have to loop through and possibly duplicate
    // std::vector<TriangleFace> left_side(faces.begin(), faces.begin() + half_size);
    // std::vector<TriangleFace> right_side(faces.begin() + half_size, faces.end());

    std::vector<TriangleFace> left_side;
    std::vector<TriangleFace> right_side;

    left_side.reserve(half_size);
    right_side.reserve(half_size);

    for (auto &face : faces) {
        auto v0 = m_vertices[face.i0];
        auto v1 = m_vertices[face.i1];
        auto v2 = m_vertices[face.i2];

        if (v0[axis] >= splitting_value ||
            v1[axis] >= splitting_value ||
            v2[axis] >= splitting_value) {
            right_side.push_back(face);
        }

        if (v0[axis] < splitting_value ||
            v1[axis] < splitting_value ||
            v2[axis] < splitting_value) {
            left_side.push_back(face);
        }
    }

    node.splitting_axis = current_axis;
    node.splitting_value = splitting_value;
    node.faces = nullptr;
    node.face_count = 0;
    cudaMallocManaged(&node.left, sizeof(TreeNode));
    cudaMallocManaged(&node.right, sizeof(TreeNode));
    build_node(*node.left, left_side, static_cast<Axis>((current_axis + 1) % 3));
    build_node(*node.right, right_side, static_cast<Axis>((current_axis + 1) % 3));
}

__device__ bool is_leaf(TreeNode *node) {
    return node->faces;
}

__device__ bool intersects_mesh(const WorldSpaceRay &ray, TreeNode *node, glm::vec3 *vertices, float &tmax) {
    auto success = false;
    for (auto i = 0; i < node->face_count; ++i) {
        auto v0 = vertices[node->faces[i].i0];
        auto v1 = vertices[node->faces[i].i1];
        auto v2 = vertices[node->faces[i].i2];

        float hit_distance = 0.0f;

        auto hit_result = hit_triangle(ray, v0, v1, v2, hit_distance);

        if (hit_result && hit_distance < tmax) {
            // tmax = hit_distance;
            success = true;
        }
    }

    return success;
}

__device__ Tuple<TreeNode *> order_subnodes(const WorldSpaceRay &ray, TreeNode *node, float tmin, float tmax) {
    auto axis = static_cast<int>(node->splitting_axis);
    auto tmin_axis = ray.origin().as_vec3()[axis] + (ray.direction()[axis] * tmin);
    auto tmax_axis = ray.origin().as_vec3()[axis] + (ray.direction()[axis] * tmax);

    if (tmin_axis < node->splitting_value && tmax_axis < node->splitting_value) {
        return {
                node->left,
                node->right
        };
    } else if (tmin_axis >= node->splitting_value && tmax_axis >= node->splitting_value) {
        return {
                node->left,
                node->right
        };
    } else if (tmin_axis < node->splitting_value && tmax_axis > node->splitting_value) {
        return {
                node->left,
                node->right
        };
    } else {
        return {
                node->right,
                node->left
        };
    }

}

enum class RangePlaneComparison {
    BelowPlane,
    AbovePlane,
    BelowToAbove,
    AboveToBelow
};

__device__ RangePlaneComparison
CompareRangeWithPlane(const WorldSpaceRay &ray, float tmin, float tmax, TreeNode *node) {
    auto axis = (int) node->splitting_axis;
    // TODO: Extract components before performing multiplication etc.
    auto range_start = ray.origin().as_vec3() + (ray.direction() * tmin);
    auto range_end = ray.origin().as_vec3() + (ray.direction() * tmax);

    auto splitting_value = node->splitting_value;

    if (range_start[axis] < splitting_value && range_end[axis] < splitting_value) {
        return RangePlaneComparison::BelowPlane;
    } else if (range_start[axis] >= splitting_value && range_end[axis] >= splitting_value) {
        return RangePlaneComparison::AbovePlane;
    } else if (range_start[axis] < splitting_value && range_end[axis] >= splitting_value) {
        return RangePlaneComparison::BelowToAbove;
    } else if (range_start[axis] >= splitting_value && range_end[axis] < splitting_value) {
        return RangePlaneComparison::AboveToBelow;
    }

    assert(false);
}


__device__ bool IndexedDeviceMesh::intersect(const WorldSpaceRay &ray, float &out_distance) {
    out_distance = FLT_MAX;
    float global_tmin = 0.0f;
    float global_tmax = 0.0f;
    if (!hit_aabb(ray, m_bounds, global_tmin, global_tmax)) {
        return false;
    }

    DeviceStack<STACK_SIZE, NodeSearchData> nodes;
    nodes.push({
                       m_root,
                       global_tmin,
                       global_tmax
               });

    /*nodes.push({
        m_root->left,
        tmin,
        tmax
    });
    nodes.push({
                       m_root->right,
                       tmin,
                       tmax
               });*/


    while (!nodes.is_empty()) {
        auto current = nodes.pop();
        auto node = current.node;
        auto tmin = current.tmin;
        auto tmax = current.tmax;

        if (is_leaf(node)) {
            if (intersects_mesh(ray, node, m_vertices, tmax)) {
                return true;
            }
        } else {
            auto a = (int) node->splitting_axis;
            auto thit = (node->splitting_value - ray.origin().as_vec3()[a]) / ray.direction()[a];

            switch (CompareRangeWithPlane(ray, tmin, tmax, node)) {
                case RangePlaneComparison::AbovePlane:
                    nodes.push(NodeSearchData{node->right, tmin, tmax});
                    break;
                case RangePlaneComparison::BelowPlane:
                    nodes.push(NodeSearchData{node->left, tmin, tmax});
                    break;
                case RangePlaneComparison::AboveToBelow:
                    nodes.push(NodeSearchData{node->right, tmin, thit});
                    nodes.push(NodeSearchData{node->left, thit, tmax});
                    break;
                case RangePlaneComparison::BelowToAbove:
                    nodes.push(NodeSearchData{node->right, thit, tmax});
                    nodes.push(NodeSearchData{node->left, tmin, thit});
                    break;
            }
        }
    }

    return false;
}


/*__device__ bool IndexedDeviceMesh::intersect(const WorldSpaceRay &ray, float &out_distance) {
    out_distance = FLT_MAX;
    float tmin = 0.0f;
    float tmax = 0.0f;
    if (!hit_aabb(ray, m_bounds, tmin, tmax)) {
        return false;
    }

    DeviceStack<NodeSearchData> nodes;
    return intersect_node(m_root, ray, tmin, tmax, nodes);
}*/

__device__ bool
intersect_split(TreeNode *node, const WorldSpaceRay &ray, float &tmin, float &tmax,
                DeviceStack<STACK_SIZE, NodeSearchData> &nodes) {
    auto a = (int) node->splitting_axis;
    auto thit = (node->splitting_value - ray.origin().as_vec3()[a]) / ray.direction()[a];
    auto nodes_to_search = order_subnodes(ray, node, tmin, tmax);

    if (thit >= tmax || thit < 0) {
        return intersect_node(nodes_to_search.item1, ray, tmin, tmax, nodes);
    } else if (thit <= tmin) {
        return intersect_node(nodes_to_search.item2, ray, tmin, tmax, nodes);
    } else {
        nodes.push({
                           nodes_to_search.item2,
                           thit,
                           tmax
                   });
        return intersect_node(nodes_to_search.item1, ray, tmin, thit, nodes);
    }
}

__device__ bool
intersect_node(TreeNode *node, const WorldSpaceRay &ray, float &tmin, float &tmax,
               DeviceStack<STACK_SIZE, NodeSearchData> &nodes) {
    if (node->faces) {
        return intersect_leaf(node, ray, tmin, tmax, nodes);
    } else {
        return intersect_split(node, ray, tmin, tmax, nodes);
    }
}

__device__ bool
intersect_leaf(TreeNode *node, const glm::vec3 *vertices, const WorldSpaceRay &ray, float &tmin, float &tmax,
               DeviceStack<STACK_SIZE, NodeSearchData> &nodes) {
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