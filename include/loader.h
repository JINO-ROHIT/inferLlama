#pragma once

#include <iostream>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <sys/mman.h>

#include "config.h"

class Tensor {
    public:
        enum class Dtype {
            FLOAT32 = 0,
            FLOAT16 = 1,
            INT32 = 2,
            INT64 = 3,
            UNKNOWN = 255
        };
    private:
        int fd;

        void* mapped_addr;
        size_t mapped_size;
        Dtype dtype;
        std::vector<int> shape;
        void* data_ptr;
    
    public:
        Tensor() : mapped_addr(nullptr), mapped_size(0), fd(-1), dtype(Dtype::UNKNOWN), data_ptr(nullptr) {}
        
        ~Tensor() {
            unmap();
        }

        template<typename T>
        const T* data() const {
            return static_cast<const T*>(data_ptr);
        }

        const std::vector<int>& get_shape() const { return shape; }
        Dtype get_dtype() const { return dtype; }

        size_t size() const {
            size_t total = 1;
            for (int dim : shape) total *= dim;
            return total;
        }

        bool load(const std::string& filename){
            fd = open(filename.c_str(), O_RDONLY);
            if(fd < 0){
                std::cerr << "Failed to open: " << filename << std::endl;
                return false;
            }

            size_t fsize = std::filesystem::file_size(filename);

            char* mapped_addr = (char*)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
            if(mapped_addr == MAP_FAILED){
                std::cerr << "Failed to mmap: " << filename << std::endl;
            }

            char* ptr = mapped_addr;

            // we know dtype is byte 0
            dtype = static_cast<Dtype>(*reinterpret_cast<uint8_t*>(ptr));
            ptr += 1;
            
            // we know dim is byte 1
            uint8_t ndims = *reinterpret_cast<uint8_t*>(ptr);
            ptr += 1;

            // Read shape
            shape.resize(ndims);
            for (int i = 0; i < ndims; ++i) {
                shape[i] = *reinterpret_cast<int*>(ptr);
                ptr += sizeof(int);
            }

            // we know data starts from byte 9
            data_ptr = ptr;

            return true;
        }

        void unmap(){
            if (mapped_addr != nullptr) {
                munmap(mapped_addr, mapped_size);
                mapped_addr = nullptr;
                data_ptr = nullptr;
            }
            if (fd != -1) {
                close(fd);
                fd = -1;
            }
        }
};

class Loader {
    private:
        std::unordered_map<std::string, Tensor> tensors;
    public:
        bool load_weights(const std::string& dir_path){
            std::cout << "loading weights from " << dir_path << std::endl;

            for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (entry.path().extension() == ".bin") {
                std::string filename = entry.path().filename().string();
                std::string tensor_name = filename.substr(0, filename.length() - 4); // the .bin part
                
                // underscores back to dots
                std::replace(tensor_name.begin(), tensor_name.end(), '_', '.');
                
                Tensor tensor;
                if (tensor.load(entry.path().string())) {
                    tensors[tensor_name] = std::move(tensor);
                    //std::cout << "Mapped: " << tensor_name << std::endl; for debug
                }
            }
            }
            return !tensors.empty();
        }

        const Tensor& get_tensor(const std::string& tensor_name) const{

            auto it = tensors.find(tensor_name);

            if(it == tensors.end()){
                std::cerr << "tensor not found";
            }

            return it->second;
        }

        bool has_tensor(const std::string& name) const {
        return tensors.find(name) != tensors.end();
        }
        
        size_t num_tensors() const {
            return tensors.size();
        }
};