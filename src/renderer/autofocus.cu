#include "autofocus.cuh"
#include "camera.cuh"
#include "scene.cuh"

__global__ void autofocus_kernel(Camera* camera, const Scene* scene, size_t width, size_t height){

    auto image_center_x = width / 2;
    auto image_center_y = height / 2;

    WorldSpaceRay ray = camera->cast_ray(image_center_x, image_center_y);

    float closest_intersection;
    if(!scene->closest_intersection(ray, closest_intersection)) {
        closest_intersection = 1000.0f;
    }

    camera->set_focal_length(closest_intersection);
}

void device_autofocus(Camera* camera, const Scene* scene, size_t width, size_t height) {
    autofocus_kernel<<<1, 1>>>(camera, scene, width, height);
    cudaDeviceSynchronize();
}