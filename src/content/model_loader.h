
#ifndef RENDERER_MODEL_LOADER_H
#define RENDERER_MODEL_LOADER_H

#include <memory>
#include <string>
#include "model.h"

class aiMesh;
class aiMaterial;

class ModelLoader {
public:
    std::unique_ptr<Model> load(const std::string& path); // TODO: Investigate why I cant use string_view here.
private:
    Mesh load_single_mesh(aiMesh* mesh);
    void load_single_material(aiMaterial *material);
};


#endif //RENDERER_MODEL_LOADER_H
