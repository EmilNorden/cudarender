//
// Created by emil on 2021-05-09.
//

#include "renderer.cuh"
#include "camera.cuh"
#include "scene.cuh"

#include "cuda_utils.cuh"
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include "device_random.cuh"
#include "transform.cuh"
#include "geometry_helpers.cuh"
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/norm.hpp> // for length2

struct EmissiveSurface {
    // incoming emission
    glm::vec3 incoming_direction{};
    glm::vec3 incoming_emission{};

    // inherent emission
    glm::vec3 emission{};

    // surface data
    glm::vec3 world_space_normal{};
    glm::vec3 world_coordinate{};
    glm::vec3 diffuse_color;
    float roughness{};
    SceneEntity *entity; // Remove this later and compare intersections using coordinates or something
};

template<size_t Length>
struct LightPath {
    EmissiveSurface surfaces[Length];
    size_t surface_count{};
};

__device__ float matte_brdf(const glm::vec3 &incoming, const ::glm::vec3 &outgoing, const glm::vec3 &surface_normal) {
    auto half = (incoming + outgoing) / 2.0f;
    auto theta = glm::dot(surface_normal, half);
    if (theta < 0) {
        return 0;
    }

    return theta;
}

__device__ glm::vec3
generate_unit_vector_in_cone(const glm::vec3 &cone_direction, float cone_angle, RandomGenerator &random) {

    // Find a tangent orthogonal to cone direction
    auto tangent = glm::vec3{1, 0, 0};
    if (glm::dot(tangent, cone_direction) > 0.99f) {
        tangent = glm::vec3{0, 0, 1};
    }
    tangent = glm::cross(cone_direction, tangent);

    // Now that we have an orthogonal tangent. Rotate it a random direction around
    // The cone direction.
    tangent = glm::rotate(tangent, random.value() * glm::two_pi<float>(), cone_direction);

    // Now, rotate cone_direction around the tangent :)
    return glm::rotate(cone_direction, (1.0f - (random.value() * 2.0f)) * cone_angle, tangent);
}

template<int N>
__device__ glm::vec3
trace_ray(const WorldSpaceRay &ray, Scene *scene, LightPath<N> &light_path, RandomGenerator &random, int depth) {
    if (depth == 0) {
        return glm::vec3(1, 1, 0);
    }

    Intersection intersection;
    SceneEntity *entity = nullptr;
    if (scene->intersect(ray, intersection, &entity)) {

        auto intersection_coordinate = ray.origin() + (ray.direction() * intersection.distance);
        auto &material = entity->mesh()->material();
        float w = 1.0f - intersection.u - intersection.v;

        auto texcoord0 = entity->mesh()->texture_coordinates()[intersection.i0];
        auto texcoord1 = entity->mesh()->texture_coordinates()[intersection.i1];
        auto texcoord2 = entity->mesh()->texture_coordinates()[intersection.i2];

        auto texture_uv = texcoord0 * w + texcoord1 * intersection.u + texcoord2 * intersection.v;

        // Calculate normal
        auto n0 = entity->mesh()->normals()[intersection.i0];
        auto n1 = entity->mesh()->normals()[intersection.i1];
        auto n2 = entity->mesh()->normals()[intersection.i2];
        auto object_space_normal = glm::normalize(n0 * w + n1 * intersection.u + n2 * intersection.v);

        if (material.has_normal_map()) {
            auto t0 = entity->mesh()->tangents()[intersection.i0];
            auto t1 = entity->mesh()->tangents()[intersection.i1];
            auto t2 = entity->mesh()->tangents()[intersection.i2];
            auto object_space_tangent = glm::normalize(t0 * w + t1 * intersection.u + t2 * intersection.v);

            auto b0 = entity->mesh()->bitangents()[intersection.i0];
            auto b1 = entity->mesh()->bitangents()[intersection.i1];
            auto b2 = entity->mesh()->bitangents()[intersection.i2];
            auto object_space_bitangent = glm::normalize(b0 * w + b1 * intersection.u + b2 * intersection.v);

            auto sampled_normal = material.sample_normal(texture_uv);
            // return sampled_normal;
            object_space_normal = glm::normalize(
                    geom::get_object_space_normal_from_normal_map(object_space_normal, sampled_normal,
                                                                  object_space_tangent, object_space_bitangent));
        }

        auto world_space_normal = entity->transpose_inverse_world().transform_normal(object_space_normal);

        auto diffuse_color = material.sample_diffuse(texture_uv);

        if (material.translucence() > 0) {
            auto theta = glm::dot(ray.direction(), world_space_normal);
            auto next_refractive_index = 1.0f;
            auto from_refractive_index = ray.refractive_index();
            auto refraction_normal = world_space_normal;
            if (theta > 0) {
                // surface normal and ray pointing in same direction. We are entering vacuum.
                refraction_normal = -refraction_normal;
            } else {
               //  return glm::vec3(1, 0, 1);
                // We are entering some medium.
                next_refractive_index = 1.333f; // TODO: MAterial should give refractive index
            }

            auto refracted_direction = glm::refract(ray.direction(), refraction_normal, from_refractive_index / next_refractive_index);

            auto refracted_ray = WorldSpaceRay{
                    intersection_coordinate + (refracted_direction * 0.1f),
                    refracted_direction,
                    next_refractive_index
            };

            auto refracted_color = trace_ray<N>(refracted_ray, scene, light_path, random, depth - 1);
            diffuse_color = refracted_color * material.translucence();
        }

        // auto reflectivity = material.reflectivity();
        // glm::vec3 reflectivityX;
        glm::vec3 reflectivity(material.reflectivity());
        if (material.has_roughness_map()) {
            reflectivity += glm::vec3(1.0) - material.sample_roughness(texture_uv);
            // reflectivity += 1.0f - material.sample_roughness(texture_uv).x;
        }

        if (reflectivity.x > 0 || reflectivity.y > 0 || reflectivity.z > 0) {
            auto reflected_direction = glm::reflect(ray.direction(), world_space_normal);
            auto cone_angle = (1.0f - reflectivity.x) * glm::pi<float>();

            reflected_direction = generate_unit_vector_in_cone(reflected_direction, cone_angle, random);

            auto reflected_ray = WorldSpaceRay{
                    intersection_coordinate + (reflected_direction * 0.05f),
                    reflected_direction
            };

            auto reflected_color = trace_ray<N>(reflected_ray, scene, light_path, random, depth - 1);
            // diffuse_color = lerp(diffuse_color, reflected_color, reflectivity);
            diffuse_color += reflected_color * (reflectivity);
        }

        glm::vec3 incoming_light{};

        for (auto i = 0; i < light_path.surface_count; ++i) {
            auto light_vector = light_path.surfaces[i].world_coordinate - intersection_coordinate;
            auto shadow_ray = WorldSpaceRay{
                    intersection_coordinate + (world_space_normal * 0.1f),
                    glm::normalize(light_vector)
            };

            Intersection shadow_intersection;
            SceneEntity *shadow_entity;
            if (scene->intersect(shadow_ray, shadow_intersection, &shadow_entity)) {
                if (shadow_entity == light_path.surfaces[i].entity) {
                    auto theta = glm::dot(shadow_ray.direction(), world_space_normal);
                    auto inv_square_law = 1.0f / glm::length2(light_vector);
                    if (theta < 0) {
                        theta = 0;
                    }
                    if (i == 0) {
                        incoming_light += light_path.surfaces[i].emission * theta * inv_square_law;
                    } else {
                        auto brdf_light = matte_brdf(light_path.surfaces[i].incoming_direction,
                                                     shadow_ray.direction() * -1.0f,
                                                     light_path.surfaces[i].world_space_normal) *
                                          light_path.surfaces[i].incoming_emission;
                        incoming_light += brdf_light * theta;
                    }
                    // incoming_light += light_path.surfaces[i].emission;


                }
            }
        }

        return material.emission() + (incoming_light * diffuse_color);
        // return material.emission() + (incoming_light * lerp(diffuse_color, reflected_color, material.reflectivity()));
    } else {
        if(scene->sky_texture() == nullptr) {
            return {0, 0, 0};
        }
        auto pitch = glm::half_pi<float>() -  glm::asin(-ray.direction().y);
        auto yaw = std::atan2(ray.direction().x, ray.direction().z);
        return scene->sky_texture()->sample({yaw, pitch / glm::pi<float>()});
    }


}

template<int N>
__device__ LightPath<N> generate_light_path(Scene *scene, RandomGenerator &random) {
    LightPath<N> result;

    auto light = scene->get_random_emissive_entity(random);
    auto surface = light->get_random_emissive_surface(random);
    /*if(surface.world_normal.y < -0.5) {
        printf("%f;%f\n", surface.world_coordinate.x, surface.world_coordinate.z);
    }*/
    /*printf("surface: %f %f %f -- %f %f %f\n", surface.world_coordinate.x, surface.world_coordinate.y, surface.world_coordinate.z,
           surface.world_normal.x, surface.world_normal.y, surface.world_normal.z);*/

    result.surfaces[0].world_space_normal = surface.world_normal;
    result.surfaces[0].world_coordinate = surface.world_coordinate;
    result.surfaces[0].emission = surface.emission; // TODO: Should this be emission in the direction of the next surface?
    result.surfaces[0].entity = light;
    result.surfaces[0].roughness = 1.0f;
    result.surfaces[0].incoming_emission = glm::vec3(0.0, 0.0, 0.0);
    result.surface_count = 1;



    for (result.surface_count = 1; result.surface_count < N; ++result.surface_count) {
        auto &previous_surface = result.surfaces[result.surface_count - 1];
        auto &current_surface = result.surfaces[result.surface_count];
        // If we just take a random unit vector in the hemisphere of the normal, we get a perfectly diffuse look on every surface
        // I think I need to calculate the reflected angle and then sample a random vector around THAT, but constrained based on roughness map



        /*if(result.surfaces[result.surface_count - 1].entity->mesh()->material().has_roughness_map()){ // Material will be moved to SceneEntity soon anyways

        }*/

        glm::vec3 path_direction = geom::random_unit_in_hemisphere(previous_surface.world_space_normal, random);

        auto path_ray = WorldSpaceRay{
                previous_surface.world_coordinate + (path_direction * 0.1f),
                path_direction
        };

        Intersection intersection;
        SceneEntity *entity;

        if (!scene->intersect(path_ray, intersection, &entity)) {
            break;
        }

        auto &material = entity->mesh()->material();
        float w = 1.0f - intersection.u - intersection.v;

        auto texcoord0 = entity->mesh()->texture_coordinates()[intersection.i0];
        auto texcoord1 = entity->mesh()->texture_coordinates()[intersection.i1];
        auto texcoord2 = entity->mesh()->texture_coordinates()[intersection.i2];

        auto texture_uv = texcoord0 * w + texcoord1 * intersection.u + texcoord2 * intersection.v;

        auto diffuse = material.sample_diffuse(texture_uv);

        // Calculate normal
        auto n0 = entity->mesh()->normals()[intersection.i0];
        auto n1 = entity->mesh()->normals()[intersection.i1];
        auto n2 = entity->mesh()->normals()[intersection.i2];
        auto object_space_normal = glm::normalize(n0 * w + n1 * intersection.u + n2 * intersection.v);

        /*if (entity->mesh()->material().has_normal_map()) {
            auto t0 = entity->mesh()->tangents()[intersection.i0];
            auto t1 = entity->mesh()->tangents()[intersection.i1];
            auto t2 = entity->mesh()->tangents()[intersection.i2];
            auto object_space_tangent = glm::normalize(t0 * w + t1 * intersection.u + t2 * intersection.v);

            auto b0 = entity->mesh()->bitangents()[intersection.i0];
            auto b1 = entity->mesh()->bitangents()[intersection.i1];
            auto b2 = entity->mesh()->bitangents()[intersection.i2];
            auto object_space_bitangent = glm::normalize(b0 * w + b1 * intersection.u + b2 * intersection.v);

            auto sampled_normal = material.sample_normal(texture_uv);
            // return sampled_normal;
            object_space_normal = glm::normalize(
                    geom::get_object_space_normal_from_normal_map(object_space_normal, sampled_normal,
                                                                  object_space_tangent, object_space_bitangent));
        }*/



        if (material.has_roughness_map()) {
            auto sampled_roughness = material.sample_roughness(texture_uv).x;
        }

        auto world_space_normal = entity->transpose_inverse_world().transform_normal(
                object_space_normal);

        auto dir_prev_surface = path_direction * -1.0f;
        current_surface.roughness = material.has_roughness_map() ? material.sample_roughness(
                texture_uv).x : 1.0f;
        current_surface.world_coordinate =
                path_ray.origin() + (intersection.distance * path_direction);
        current_surface.world_space_normal = world_space_normal;
        current_surface.emission = material.emission();

        current_surface.incoming_direction = dir_prev_surface;
        if (result.surface_count == 1) {
            // Previous surface IS the light source
            current_surface.incoming_emission = previous_surface.emission;
        } else {
            auto amount_light_reflected = matte_brdf(
                    previous_surface.incoming_direction,
                    path_direction,
                    previous_surface.world_space_normal
            );
            current_surface.incoming_emission =
                    (amount_light_reflected * previous_surface.incoming_emission)
                    + previous_surface.emission;
        }
        current_surface.incoming_emission *= diffuse;
        current_surface.entity = entity;
    }

    return result;
}

__global__ void
cudaRender(float *g_odata, Camera *camera, Scene *scene, RandomGeneratorPool *random_pool, int width, int height,
           size_t sample) {
    constexpr int PathLength = 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;

    auto threads_per_block = bw * bh;
    auto thread_num_in_block = tx + bw * ty;
    auto block_num_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
    auto global_thread_id = block_num_in_grid * threads_per_block + thread_num_in_block;
    // auto global_block_id = block_num_in_grid;
    auto random = random_pool->get_generator(global_thread_id);

    if (x < width && y < height) {
        auto ray = camera->cast_perturbed_ray(x, y, random);

        auto light_path = generate_light_path<PathLength>(scene, random);

        auto color = trace_ray<PathLength>(ray, scene, light_path, random, 2);

        color = glm::clamp(color, {0, 0, 0}, {1, 1, 1});

        glm::vec3 previous_color;
        auto pixel_index = y * (width * 4) + (x * 4);
        g_odata[pixel_index] = ((g_odata[pixel_index] * (float) sample) + color.x) / (sample + 1.0f);
        g_odata[pixel_index + 1] = ((g_odata[pixel_index + 1] * (float) sample) + color.y) / (sample + 1.0f);
        g_odata[pixel_index + 2] = ((g_odata[pixel_index + 2] * (float) sample) + color.z) / (sample + 1.0f);
        g_odata[pixel_index + 3] = 1.0;
    }

}

void Renderer::render(Camera *camera, Scene *scene, RandomGeneratorPool *random, int width, int height, size_t sample) {
    dim3 block(16, 16, 1);
    dim3 grid(std::ceil(width / (float) block.x), std::ceil(height / (float) block.y), 1);
    cudaRender<<<grid, block, 0>>>((float *) m_cuda_render_buffer, camera, scene, random, width, height, sample);

    cudaArray *texture_ptr;
    cuda_assert(cudaGraphicsMapResources(1, &m_cuda_tex_resource, 0));
    cuda_assert(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_cuda_tex_resource, 0, 0));

    // TODO: Havent we already calculated this?
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLfloat) * num_values;
    cuda_assert(cudaMemcpyToArray(texture_ptr, 0, 0, m_cuda_render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
    cuda_assert(cudaGraphicsUnmapResources(1, &m_cuda_tex_resource, 0));
}

Renderer::Renderer(GLuint gl_texture, int width, int height)
        : m_cuda_render_buffer(nullptr) {
    allocate_render_buffer(width, height);

    cuda_assert(cudaGraphicsGLRegisterImage(&m_cuda_tex_resource, gl_texture, GL_TEXTURE_2D,
                                            cudaGraphicsRegisterFlagsWriteDiscard));
}

void Renderer::allocate_render_buffer(int width, int height) {
    if (m_cuda_render_buffer) {
        cudaFree(m_cuda_render_buffer);
    }

    auto buffer_size = width * height * 4 * sizeof(GLfloat); // Is GLubyte ever larger than 1?
    cuda_assert(cudaMalloc(&m_cuda_render_buffer, buffer_size));
}

Renderer::~Renderer() {
    if (m_cuda_render_buffer) {
        cudaFree(m_cuda_render_buffer);
    }
}

void Renderer::render(int width, int height, const Camera &camera, const Scene &scene) {

}