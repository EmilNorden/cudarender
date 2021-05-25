#ifndef RENDERER_INTERSECTION_CUH
#define RENDERER_INTERSECTION_CUH

struct Intersection {
    Intersection() = default;
    int i0{};
    int i1{};
    int i2{};
    float u{};
    float v{};
    float distance{};
};

#endif