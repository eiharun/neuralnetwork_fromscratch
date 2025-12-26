#include "neural_network.h"
#include <dataset.h>
#include <fstream>
#include <iostream>
#include <utility>

MNIST::MNIST(std::string data_path, std::string labels_path){
    load_images(data_path);
    load_labels(labels_path);
}

void MNIST::load_images(std::string data_path){
    std::ifstream file(data_path, std::ios::in | std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Unable to open file " << data_path << '\n';
        return;
    }
    unsigned int magic_number{};
    // Read four bytes
    char bytes[4];
    if(!file.read(bytes, 4)){
        std::cerr << "Unable to read magic number\n";
        return;
    }
    magic_number =  (unsigned int)bytes[0] << 24 | 
                    (unsigned int)bytes[1] << 16 | 
                    (unsigned int)bytes[2] << 8  | 
                    (unsigned int)bytes[3];
    
    char dim = (magic_number & 0xFF);
    m_type = static_cast<MNIST::type>((magic_number & 0xFF00)>>8);
    m_dims.resize(dim);
    for(int i{}; i<dim; ++i){
        if(!file.read(bytes, 4)){
            std::cerr << "Unable to read bytes\n";
            return;
        }
        // Use type here
        m_dims[i] = (unsigned char)bytes[0] << 24 | 
                    (unsigned char)bytes[1] << 16 | 
                    (unsigned char)bytes[2] << 8  | 
                    (unsigned char)bytes[3];
    }
    vector<size_t> size {m_dims[m_dims.size()-1], m_dims[m_dims.size()-2]};
    size_t tot_size = size[0] * size[1];
    size_t count{};
    std::cout << "Image -----";
    // while(count < m_dims[0]){
    while(!file.eof()){
        char byte;
        vector<float> data;
        std::cout << "\b\b\b\b\b\b" << count ;
        for(int i{}; i<tot_size; ++i){
            if(!file.get(byte)){
                std::cerr << "Unable to read image data\n";
                return;
            }
            data.push_back(static_cast<float>(static_cast<unsigned char>(byte))/255.0f);
        }
        m_data.push_back(std::make_shared<Tensor<float>>(data));
        count++;
    }
    std::cout << "Image " << count << '\n';

    file.close();
}

void MNIST::load_labels(std::string label_path){
    std::ifstream file(label_path, std::ios::in | std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Unable to open file " << label_path << '\n';
        return;
    }
    unsigned int magic_number{};
    // Read four bytes
    char bytes[4];
    if(!file.read(bytes, 4)){
        std::cerr << "Unable to read magic number\n";
        return;
    }
    magic_number =  (unsigned int)bytes[0] << 24 | 
                    (unsigned int)bytes[1] << 16 | 
                    (unsigned int)bytes[2] << 8  | 
                    (unsigned int)bytes[3];
    
    char dim = (magic_number & 0xFF);
    m_type = static_cast<MNIST::type>((magic_number & 0xFF00)>>8);
    LIB_ASSERT(dim == 1, "Label dimension must be 1");
    for(int i{}; i<dim; ++i){
        if(!file.read(bytes, 4)){
            std::cerr << "Unable to read bytes\n";
            return;
        }
    }
    while(!file.eof()){
        char byte;
        if(!file.get(byte)){
            std::cerr << "Unable to read label data\n";
            return;
        }
        m_labels.push_back(static_cast<unsigned char>(byte));
    }
    file.close();
}

int MNIST::get_length(){
    return m_labels.size();
}

std::pair<int, std::shared_ptr<Tensor<float>>> MNIST::get_item(int index){
    return std::make_pair(m_labels[index], m_data[index]);
}


void MNIST::print(int idx){
    Tensor<float> img = *m_data[idx];
    std::cout << "Label: " << (int)m_labels[idx] << "\n\n";
    for (int i = 0; i < img.cols(); i++) {
        int val = static_cast<int>(img.at(0, i) * 255); 
        // \x1b[48;2;R;G;Bm sets the Background color
        std::cout << "\x1b[48;2;" << val << ";" << val << ";" << val << "m  ";
        if((i+1)%28 == 0){
            std::cout << "\x1b[0m\n";
        }
        // \x1b[0m resets the color at the end of each row
        // std::cout << "\x1b[0m\n";
    }
}