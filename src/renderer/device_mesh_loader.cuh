#ifndef DEVICE_MESH_LOADER_CUH_
#define DEVICE_MESH_LOADER_CUH_

#include "device_mesh.cuh"
#include <vector>
#include <string>

class aiMesh;
class aiMaterial;

class DeviceMeshLoader {
public:
    __host__ std::vector<IndexedDeviceMesh> load(const std::string& path);
private:
};

#endif