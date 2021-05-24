
#include "model_loader.h"
#include "model.h"
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
#include <iostream>
#include <vector>


using namespace std;

unique_ptr<Model> ModelLoader::load(const string& path)
{
    std::cout << "Loading model " << path << std::endl;

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
        return nullptr;
    }

    // TODO: Load materials
    for(auto i = 0; i < scene->mNumMaterials; ++i) {
        load_single_material(scene->mMaterials[i]);
    }

    vector<Mesh> meshes;
    for(auto i = 0; i < scene->mNumMeshes; ++i) {
        std::cout << "  Loading mesh " << i << " (" << scene->mMeshes[i]->mName.C_Str() << ")" << std::endl;
        meshes.push_back(load_single_mesh(scene->mMeshes[i]));
        std::cout << "    Vertices: " << meshes.back().vertices().size() << "\t\t Faces: " << meshes.back().indices().size() / 3 << std::endl;
    }

    std::cout << std::endl;

    return std::make_unique<Model>(meshes);
}

Mesh ModelLoader::load_single_mesh(aiMesh* mesh) {
    std::vector<glm::vec3> vertices;
    //std::vector<tri_index> indices;
    std::vector<int> indices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> tangents;
    std::vector<glm::vec3> bitangents;
    std::vector<glm::vec2> texture_coords;

    for (size_t i = 0; i < mesh->mNumVertices; ++i) {
        aiVector3D& vertex = mesh->mVertices[i];

        vertices.push_back(glm::vec3(vertex.x, vertex.y, vertex.z) * 0.004f);

        aiVector3D& normal = mesh->mNormals[i];
        normals.push_back(glm::vec3(normal.x, normal.y, normal.z));

        if (mesh->HasTangentsAndBitangents()) {
            aiVector3D& tangent = mesh->mTangents[i];
            tangents.push_back(glm::vec3(tangent.x, tangent.y, tangent.z));

            aiVector3D& bitangent = mesh->mBitangents[i];
            bitangents.push_back(glm::vec3(bitangent.x, bitangent.y, bitangent.z));
        }
        if (mesh->HasTextureCoords(0)) {
            aiVector3D& texture_coord = mesh->mTextureCoords[0][i];
            texture_coords.push_back(glm::vec2(texture_coord.x, texture_coord.y));
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

    return Mesh{vertices, indices};
}

void ModelLoader::load_single_material(aiMaterial *material) {
    aiString name;
    aiGetMaterialString(material, AI_MATKEY_NAME, &name);

    // std::string cache_key = model_path + "_" + std::string(name.C_Str());

    // if(allow_cached_materials) {

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

    // std::shared_ptr<Texture> diffuseTexture = nullptr;
    if (material->GetTexture(aiTextureType_DIFFUSE, 0, &p) == AI_SUCCESS) {
        std::cout << "stringiii: " << p.C_Str() << std::endl;
        // diffuseTexture = load_texture(p, model_path);
    }

    // std::shared_ptr<Texture> normalTexture = nullptr;
    if (material->GetTexture(aiTextureType_NORMALS, 0, &p) == AI_SUCCESS) {
        std::cout << "WHAAATTTTTTTT\n";
        // normalTexture = load_texture(p, model_path);
    }
    /*auto mat = new Material(
            glm::vec3(diffuse_color.r, diffuse_color.g, diffuse_color.b),
            glm::vec3(emissive_color.r, emissive_color.g, emissive_color.b),
            diffuseTexture, normalTexture);*/

    // m_material_cache.insert(cache_key, mat);

   // return mat;
}

/*Material ModelLoader::load_single_material(aiMaterial* material,
                                            const std::string& model_path,
                                            bool allow_cached_materials) {
    aiString name;
    aiGetMaterialString(material, AI_MATKEY_NAME, &name);

    std::string cache_key = model_path + "_" + std::string(name.C_Str());

    // if(allow_cached_materials) {

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

    std::shared_ptr<Texture> diffuseTexture = nullptr;
    if (material->GetTexture(aiTextureType_DIFFUSE, 0, &p) == AI_SUCCESS) {
        diffuseTexture = load_texture(p, model_path);
    }

    std::shared_ptr<Texture> normalTexture = nullptr;
    if (material->GetTexture(aiTextureType_NORMALS, 0, &p) == AI_SUCCESS) {
        std::cout << "WHAAATTTTTTTT\n";
        normalTexture = load_texture(p, model_path);
    }
    auto mat = new Material(
            glm::vec3(diffuse_color.r, diffuse_color.g, diffuse_color.b),
            glm::vec3(emissive_color.r, emissive_color.g, emissive_color.b),
            diffuseTexture, normalTexture);

    // m_material_cache.insert(cache_key, mat);

    return mat;
}*/