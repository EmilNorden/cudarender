#include "device_mesh_loader.cuh"

#include "device_texture_loader.cuh"
#include "cuda_utils.cuh"

#include <filesystem>
#include <glm/glm.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
#include <iostream>

using namespace std;

__host__ IndexedDeviceMesh* load_single_mesh(aiMesh* mesh, const vector<DeviceMaterial> &materials);
__host__ DeviceMaterial load_single_material(aiMaterial *material, const filesystem::path& model_directory);

__host__ vector<IndexedDeviceMesh*> DeviceMeshLoader::load(const string& path) {
    cout << "Loading model " << path << endl;

    vector<IndexedDeviceMesh*> meshes;
    Assimp::Importer importer;

    unsigned int assimp_flags =
            aiProcess_CalcTangentSpace | aiProcess_Triangulate |
            aiProcess_GenSmoothNormals | aiProcess_ImproveCacheLocality |
            aiProcess_RemoveRedundantMaterials | aiProcess_OptimizeMeshes |
            aiProcess_FixInfacingNormals | aiProcess_FindInvalidData |
            aiProcess_OptimizeGraph | aiProcess_JoinIdenticalVertices |
            aiProcess_FindInstances | aiProcess_GenUVCoords | aiProcess_SortByPType;



    auto scene = importer.ReadFile(path, assimp_flags);
    if (!scene) {
        cerr << "Failed to load model!\n" << importer.GetErrorString() << "\n";
        return meshes;
    }

    auto model_directory = std::filesystem::path {path}.parent_path();

    // TODO: Load materials
    vector<DeviceMaterial> materials;
    for(auto i = 0; i < scene->mNumMaterials; ++i) {
        materials.push_back(load_single_material(scene->mMaterials[i], model_directory));
    }


    for(auto i = 0; i < scene->mNumMeshes; ++i) {
        cout << "  Loading mesh " << i << " (" << scene->mMeshes[i]->mName.C_Str() << ")" << endl;
        meshes.push_back(load_single_mesh(scene->mMeshes[i], materials));
    }

    cout << endl;

    return meshes;
}

__host__ IndexedDeviceMesh* load_single_mesh(aiMesh* mesh, const vector<DeviceMaterial> &materials) {
    std::vector<glm::vec3> vertices;
    std::vector<int> indices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> tangents;
    std::vector<glm::vec3> bitangents;
    std::vector<glm::vec2> texture_coords;

    for (size_t i = 0; i < mesh->mNumVertices; ++i) {
        aiVector3D& vertex = mesh->mVertices[i];

        vertices.emplace_back(vertex.x, vertex.y, vertex.z);

        aiVector3D& normal = mesh->mNormals[i];
        normals.emplace_back(normal.x, normal.y, normal.z);

        if (mesh->HasTangentsAndBitangents()) {
            aiVector3D& tangent = mesh->mTangents[i];
            tangents.emplace_back(tangent.x, tangent.y, tangent.z);

            aiVector3D& bitangent = mesh->mBitangents[i];
            bitangents.emplace_back(bitangent.x, bitangent.y, bitangent.z);
        }
        if (mesh->HasTextureCoords(0)) {
            aiVector3D& texture_coord = mesh->mTextureCoords[0][i];
            texture_coords.emplace_back(texture_coord.x, texture_coord.y);
        }
    }

    for (size_t i = 0; i < mesh->mNumFaces; ++i) {
        /*indices.push_back({mesh->mFaces[i].mIndices[0], mesh->mFaces[i].mIndices[1],
                           mesh->mFaces[i].mIndices[2]});*/
        indices.push_back(mesh->mFaces[i].mIndices[0]);
        indices.push_back(mesh->mFaces[i].mIndices[1]);
        indices.push_back(mesh->mFaces[i].mIndices[2]);
    }

    /*auto iterator = materials.find(mesh->mMaterialIndex);

    Material* material = nullptr;

    if (iterator != materials.end())
        material = iterator->second;*/

    if (texture_coords.empty()) {
        texture_coords = std::vector<glm::vec2>(vertices.size(), glm::vec2{0, 0});
    }

    /*return std::make_unique<OctreeMesh>(std::string(mesh->mName.C_Str()),
                                        vertices, indices, normals, tangents,
                                        bitangents, texture_coords, material);*/

    auto faces_from_indices = [](const std::vector<int>& indices) {
        std::vector<TriangleFace> faces;
        for(int i = 0; i < indices.size(); i += 3) {
            faces.push_back({indices[i], indices[i+1], indices[i + 2]});
        }

        return faces;
    };

    cout << "    Vertices: " << vertices.size() << "\t\t Faces: " << indices.size() / 3 << endl;

    auto& material = materials[mesh->mMaterialIndex];

    // TODO: An idea for later. Perhaps I can allocate space for all meshes at once, and return a single pointer to all instead of a vector of pointers
    IndexedDeviceMesh *device_mesh;
    cuda_assert(cudaMallocManaged(&device_mesh, sizeof(IndexedDeviceMesh)));
    new(device_mesh) IndexedDeviceMesh{vertices, normals, tangents, bitangents, faces_from_indices(indices), texture_coords, material};

    cout << "    Bounds: " << device_mesh->bounds();
    return device_mesh;
}

__host__ DeviceMaterial load_single_material(aiMaterial *material, const filesystem::path& model_directory) {
    aiString name;
    aiGetMaterialString(material, AI_MATKEY_NAME, &name);

    // std::string cache_key = model_path + "_" + std::string(name.C_Str());

    // if(allow_cached_materials) {namespace fs = std::filesystem;

    // 	auto cached_material = m_material_cache.get(cache_key);
    // 	if(cached_material)
    // 		return cached_material;

    // }

    // material->Get does NOT return colors as their documentation states.
    // aiGetMaterialColor does however. Perhaps the same problem exists with
    // Textures?

    aiColor4D fooo;
    aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &fooo);

    float refracti;
    aiGetMaterialFloat(material, AI_MATKEY_REFRACTI, &refracti);
    // material->Get<float>(AI_MATKEY_REFRACTI, refracti);

    float opacity;
    aiGetMaterialFloat(material, AI_MATKEY_OPACITY, &opacity);
    // material->Get<float>(AI_MATKEY_OPACITY, opacity);

    float shininess;
    aiGetMaterialFloat(material, AI_MATKEY_SHININESS, &shininess);
    // material->Get<float>(AI_MATKEY_SHININESS, shininess);

    int shading_model;
    aiGetMaterialInteger(material, AI_MATKEY_SHADING_MODEL, &shading_model);
    // material->Get<int>(AI_MATKEY_SHADING_MODEL, shading_model);

    aiColor4D ambient_color;
    aiGetMaterialColor(material, AI_MATKEY_COLOR_AMBIENT, &ambient_color);

    aiColor4D diffuse_color;
    aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &diffuse_color);

    aiColor4D emissive_color;
    aiGetMaterialColor(material, AI_MATKEY_COLOR_EMISSIVE, &emissive_color);

    aiColor4D specular_color;
    aiGetMaterialColor(material, AI_MATKEY_COLOR_SPECULAR, &specular_color);

    aiString p;
    unsigned int texCount = 0;
    for (int i = 0; i < aiTextureType_UNKNOWN; ++i) {
        texCount += material->GetTextureCount((aiTextureType)i);
    }

    DeviceTexture *diffuse_texture = nullptr;
    if (material->GetTexture(aiTextureType_DIFFUSE, 0, &p) == AI_SUCCESS) {

        auto texture_path = model_directory;

        texture_path /= p.C_Str();

        diffuse_texture = DeviceTextureLoader{}.load(texture_path.string()); // TODO: Inject DeviceTextureLoader
        // diffuseTexture = load_texture(p, model_path);
    }

    // std::shared_ptr<Texture> normalTexture = nullptr;
    if (material->GetTexture(aiTextureType_NORMALS, 0, &p) == AI_SUCCESS) {
        std::cout << "WHAAATTTTTTTT\n";
        // normalTexture = load_texture(p, model_path);
    }

    return DeviceMaterial{diffuse_texture};
}