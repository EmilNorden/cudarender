//
// Created by emil on 2021-05-03.
//

#include "renderer/device_stack.cuh"

int main()
{
    DeviceStack<20, int> stack;

    stack.push(123);

    stack.push(456);

    stack.push(789);

    auto a = stack.pop();
    auto b = stack.pop();
    auto c = stack.pop();
    auto d = stack.pop();
    return 0;
}