#ifndef RENDERER_KDTREE_CUH_
#define RENDERER_KDTREE_CUH_

#include <glm/glm.hpp>
#include <functional>
#include <vector>

enum Axis {
    X, Y, Z
};

template<typename Item>
class KdTree {
    using host_items = std::vector<Item>;
public:
    void build(const host_items &items, const std::function<void(host_items &, Axis)> sort_callback,
               const std::function<glm::vec3(Item &)> position_callback) {
        auto items_copy = items;
        build_node(m_root, Axis::X, items_copy, sort_callback, position_callback);
    }

private:
    struct TreeNode {
        Axis plane_axis;
        glm::vec3 plane_position;
        TreeNode *left;
        TreeNode *right;
        Item *items;
        int item_count;
    };

    TreeNode m_root;

    void build_node(TreeNode &node, Axis current_axis, host_items &items,
                    const std::function<void(host_items &, Axis)> sort_callback,
                    const std::function<glm::vec3(Item &)> position_callback) {
        if (items.size() < 100) {
            cudaMalloc(&node.items, sizeof(Item) * items.size());
            cudaMemcpy(node.items, items.data(), sizeof(Item) * items.size(), cudaMemcpyHostToDevice);
            node.item_count = items.size();
            return;
        }

        node.plane_axis = current_axis;
        sort_callback(items, current_axis);

        auto half_size = items.size() / 2;
        std::vector<Item> left_side(items.begin(), items.begin() + half_size);
        std::vector<Item> right_side(items.begin() + half_size, items.end());

        auto median = position_callback(left_side.back());
        cudaMallocManaged(&node.left, sizeof(TreeNode));
        cudaMallocManaged(&node.right, sizeof(TreeNode));

        build_node(*node.left, static_cast<Axis>((current_axis + 1) % 3),left_side, sort_callback, position_callback);



    }
};

#endif