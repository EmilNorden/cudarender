#include "device_mesh.cuh"
#include "device_stack.cuh" // For kd tree traversal
#include "device_random.cuh"
#include "cuda_utils.cuh"

using namespace std;

#define STACK_SIZE      20

struct TreeNode {
    Axis splitting_axis;
    float splitting_value;
    TreeNode *left;
    TreeNode *right;
    TriangleFace *faces;
    size_t face_count;
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

#define EPSILON 9.99999997475243E-07


__device__ bool hit_aabb(const ObjectSpaceRay &ray, const AABB &aabb, float &out_near, float &out_far) {
    out_near = FLT_MIN;
    out_far = FLT_MAX;
    for (auto i = 0; i < 3; ++i) {
        /*if glm::abs(self.direction[i]) < EPSILON {
            if self.origin[i] < bounds.min()[i] as f32 || self.origin[i] > bounds.max()[i] as f32 {
                return false
            }
        }
         */
        auto t1 = (aabb.min()[i] - ray.origin()[i]) * ray.inverse_direction()[i];// / self.direction[i];
        auto t2 = (aabb.max()[i] - ray.origin()[i]) * ray.inverse_direction()[i]; // / self.direction[i];

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

__device__ bool
hit_triangle(const ObjectSpaceRay &ray, const glm::vec3 *vertices, TriangleFace &face, float &out_u, float &out_v,
             float &out_distance) {
    auto v1 = vertices[face.i0];

    glm::vec3 e1 = face.e1;
    glm::vec3 e2 = face.e2;

    // Begin calculating determinant - also used to calculate u parameter
    glm::vec3 P = glm::cross(ray.direction(), e2); // m_direction.cross(e2);
    // if determinant is near zero, ray lies in plane of triangle

    float det = glm::dot(e1, P); // e1.dot(P);

    /*if (det > -EPSILON && det < EPSILON)
        return false;*/

    // BACK-FACE CULLING

    /*if (det < 0.00000001f) {
        return false;
    }*/
    /*if (!inside_geometry && det < EPSILON) {
        return false;
    }*/

    float inv_det = 1.0f / det;

    // calculate distance from V1 to ray origin
    glm::vec3 T = ray.origin() - v1;

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
        out_u = u;
        out_v = v;
        return true;
    }

    // No hit, no win
    return false;
}


__host__ IndexedDeviceMesh::IndexedDeviceMesh(const std::vector<glm::vec3> &vertices,
                                              const std::vector<glm::vec3> &normals,
                                              const std::vector<glm::vec3> &tangents,
                                              const std::vector<glm::vec3> &bitangents,
                                              const std::vector<TriangleFace> &faces,
                                              const std::vector<glm::vec2> &tex_coords,
                                              const DeviceMaterial &material)
        : m_bounds(vertices, 0.01f), m_material(material) {

    cuda_assert(cudaMallocManaged(&m_vertices, sizeof(glm::vec3) * vertices.size()));
    cuda_assert(cudaMemcpy(m_vertices, vertices.data(), sizeof(glm::vec3) * vertices.size(), cudaMemcpyHostToDevice));

    transfer_vector_to_device_memory(normals, &m_normals);
    transfer_vector_to_device_memory(tangents, &m_tangents);
    transfer_vector_to_device_memory(bitangents, &m_bitangents);
    transfer_vector_to_device_memory(tex_coords, &m_tex_coords);

    cuda_assert(cudaMallocManaged(&m_root, sizeof(TreeNode)));
    auto faces_copy = faces;
    build_node(*m_root, faces_copy, Axis::X);
}

void IndexedDeviceMesh::build_node(TreeNode &node, std::vector<TriangleFace> &faces, Axis current_axis) {

    if (faces.size() < 100) {
        cuda_assert(cudaMalloc(&node.faces, sizeof(TriangleFace) * faces.size()));
        cuda_assert(cudaMemcpy(node.faces, faces.data(), sizeof(TriangleFace) * faces.size(), cudaMemcpyHostToDevice));
        node.face_count = faces.size();
        node.left = nullptr;
        node.right = nullptr;
        return;
    }

    auto axis = static_cast<int>(current_axis);

    std::sort(faces.begin(), faces.end(), [&](const TriangleFace &a, const TriangleFace &b) {
        // Sort each face by comparing the center of the triangles.
        // Previously I used the first vertex of each face but that didnt work out well.

        auto a_mid_point = (m_vertices[a.i0] + m_vertices[a.i1] + m_vertices[a.i2]) / 3.0f;
        auto b_mid_point = (m_vertices[b.i0] + m_vertices[b.i1] + m_vertices[b.i2]) / 3.0f;

        return a_mid_point[axis] < b_mid_point[axis];
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

    for (auto &face: faces) {
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
    cuda_assert(cudaMallocManaged(&node.left, sizeof(TreeNode)));
    cuda_assert(cudaMallocManaged(&node.right, sizeof(TreeNode)));
    build_node(*node.left, left_side, static_cast<Axis>((current_axis + 1) % 3));
    build_node(*node.right, right_side, static_cast<Axis>((current_axis + 1) % 3));
}

__device__ bool is_leaf(TreeNode *node) {
    return node->faces;
}

__device__ bool
intersects_mesh(const ObjectSpaceRay &ray, TreeNode *node, glm::vec3 *vertices, Intersection &intersection,
                float &tmax) {
    auto success = false;
    for (auto i = 0; i < node->face_count; ++i) {
        auto i0 = node->faces[i].i0;
        auto i1 = node->faces[i].i1;
        auto i2 = node->faces[i].i2;
        auto v0 = vertices[i0];
        auto v1 = vertices[i1];
        auto v2 = vertices[i2];

        float hit_distance = 0.0f;

        float u = 0.0f;
        float v = 0.0f;
        auto hit_result = hit_triangle(ray, vertices, node->faces[i], u, v, hit_distance);

        if (hit_result && hit_distance < tmax) {
            tmax = hit_distance;
            intersection.i0 = i0;
            intersection.i1 = i1;
            intersection.i2 = i2;
            intersection.u = u;
            intersection.v = v;
            success = true;
        }
    }

    return success;
}

enum class RangePlaneComparison {
    BelowPlane,
    AbovePlane,
    BelowToAbove,
    AboveToBelow
};

__device__ RangePlaneComparison
CompareRangeWithPlane(const ObjectSpaceRay &ray, float tmin, float tmax, TreeNode *node) {
    auto axis = (int) node->splitting_axis;
    // TODO: Extract components before performing multiplication etc.
    auto range_start = ray.origin()[axis] + (ray.direction()[axis] * tmin);
    auto range_end = ray.origin()[axis] + (ray.direction()[axis] * tmax);

    auto splitting_value = node->splitting_value;

    if (range_start < splitting_value && range_end < splitting_value) {
        return RangePlaneComparison::BelowPlane;
    } else if (range_start >= splitting_value && range_end >= splitting_value) {
        return RangePlaneComparison::AbovePlane;
    } else if (range_start < splitting_value && range_end >= splitting_value) {
        return RangePlaneComparison::BelowToAbove;
    } else if (range_start >= splitting_value && range_end < splitting_value) {
        return RangePlaneComparison::AboveToBelow;
    }

    assert(false);
}

__device__ bool
IndexedDeviceMesh::intersect(const ObjectSpaceRay &ray, Intersection &intersection) {
    intersection.distance = FLT_MAX;
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

    while (!nodes.is_empty()) {
        auto current = nodes.pop();
        auto node = current.node;
        auto tmin = current.tmin;
        auto tmax = current.tmax;

        if (is_leaf(node)) {
            if (intersects_mesh(ray, node, m_vertices, intersection, tmax)) {
                intersection.distance = tmax;
                return true;
            }
        } else {
            auto a = (int) node->splitting_axis;
            auto thit = (node->splitting_value - ray.origin()[a]) * ray.inverse_direction()[a];

            switch (CompareRangeWithPlane(ray, tmin, tmax, node)) {
                case RangePlaneComparison::AbovePlane:
                    nodes.push(NodeSearchData{node->right, tmin, tmax});
                    break;
                case RangePlaneComparison::BelowPlane:
                    nodes.push(NodeSearchData{node->left, tmin, tmax});
                    break;
                case RangePlaneComparison::AboveToBelow:
                    nodes.push(NodeSearchData{node->left, thit, tmax});
                    nodes.push(NodeSearchData{node->right, tmin, thit});

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

[[nodiscard]] __device__ TriangleFace IndexedDeviceMesh::get_random_face(RandomGenerator &random) {
    TreeNode *current = m_root;

    while (!is_leaf(current)) {
        if (random.value() >= 0.5f) {
            current = current->left;
        } else {
            current = current->right;
        }
    }

    auto face_index = static_cast<size_t>(random.value() * current->face_count);


    return current->faces[face_index];
}