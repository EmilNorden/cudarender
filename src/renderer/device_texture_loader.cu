#include "device_texture_loader.cuh"

#include "device_texture.cuh"

#include <FreeImage.h>
#include <vector>
#include <filesystem>
#include <iostream>

using namespace std;

DeviceTexture *DeviceTextureLoader::load(const std::string &path) {
    cout << "Loading texture " << path << endl;
    if(!filesystem::exists(path)) {
        cerr << "  File does not exist!" << endl;
        exit(1);
    }

    FREE_IMAGE_FORMAT type = FreeImage_GetFileType(path.c_str());

    if (type == FIF_UNKNOWN) {
        type = FreeImage_GetFIFFromFilename(path.c_str());

        if (type == FIF_UNKNOWN){
            cerr << "  Unable to determine texture format!" << endl;
            exit(1);
        }
    }

    FIBITMAP *bitmap = FreeImage_Load(type, path.c_str());
    auto width = FreeImage_GetWidth(bitmap);
    auto height = FreeImage_GetHeight(bitmap);
    std::vector <uint8_t> pixels;
    pixels.reserve(3 * width * height);
    for (auto y = 0; y < height; ++y) {
        for (auto x = 0; x < width; ++x) {
            RGBQUAD pixel;
            FreeImage_GetPixelColor(bitmap, x, y, &pixel);
            pixels.push_back(pixel.rgbRed);
            pixels.push_back(pixel.rgbGreen);
            pixels.push_back(pixel.rgbBlue);
        }
    }

    FreeImage_Unload(bitmap);

    cout << endl;

    DeviceTexture *texture;
    cudaMallocManaged(&texture, sizeof(DeviceTexture));
    return new(texture) DeviceTexture{pixels, width, height};
}

bool DeviceTextureLoader::file_is_supported(const std::string &path) {
    FREE_IMAGE_FORMAT type = FreeImage_GetFileType(path.c_str());

    if (type == FIF_UNKNOWN) {
        type = FreeImage_GetFIFFromFilename(path.c_str());
    }

    return type != FIF_UNKNOWN;
}