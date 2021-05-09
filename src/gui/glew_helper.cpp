//
// Created by emil on 2021-05-09.
//

#include "glew_helper.h"
#include <GL/glew.h>
#include <iostream>

bool GlewHelper::m_initialized = false;

void GlewHelper::init() {
    if(m_initialized) {
        return;
    }

    m_initialized = true;
    glewExperimental = GL_TRUE; // need this to enforce core profile
    GLenum err = glewInit();
    glGetError();
    if(err != GLEW_OK) {
        std::cerr << "glewInit failed: " << glewGetErrorString(err) << std::endl;
        exit(1);
    }
    // check_for_gl_errors();
}