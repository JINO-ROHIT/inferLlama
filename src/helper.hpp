#pragma once
#include <chrono>
#include <iostream>
#include <string>

//TO-DO: find a nicer way to do this?
class Timer {
public:
    Timer(const std::string& name = "Timer")
        : name(name), start(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = end - start;

        auto sec  = std::chrono::duration<double>(diff).count();
        auto ms   = std::chrono::duration<double, std::milli>(diff).count();

        std::cout << name << ":\n"
                  << "  " << sec << " seconds\n"
                  << "  " << ms  << " ms\n";
    };

private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
};
