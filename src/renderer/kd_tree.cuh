#ifndef RENDERER_KDTREE_CUH_
#define RENDERER_KDTREE_CUH_

#include <glm/glm.hpp>
#include <functional>

enum Axis {
    X, Y, Z
};

template <typename Item>
class KdTree{
    using host_items = thrust::host_vector<Item>;
public:
    explicit __host__ KdTree(const host_items& items, const std::function<void(host_items&, Axis)> sort_callback) {
        Axis current_axis = Axis::X;
        // Sortera items på X axis och ta ut median
        // Behöver vi en SortByAxis func?
    }
private:
    struct TreeNode {
        Axis plane_axis;
        glm::vec3 plane_position;
        TreeNode *left;
        TreeNode *right;
        thrust::device_vector<Item> items;
    };

    TreeNode root;
};

#endif