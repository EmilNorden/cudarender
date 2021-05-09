//
// Created by emil on 2021-05-09.
//

#ifndef RENDERER_GLEW_HELPER_H
#define RENDERER_GLEW_HELPER_H


class GlewHelper {
public:
    static void init();
private:
    static bool m_initialized;
};


#endif //RENDERER_GLEW_HELPER_H
