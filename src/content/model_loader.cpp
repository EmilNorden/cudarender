
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