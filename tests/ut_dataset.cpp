#include <gtest/gtest.h>
#include <dataset.h>

TEST(LoadImage, LoadAndPrint){
    const std::string data_path {"/home/harun/Documents/neural_network/train-images.idx3-ubyte"};
    const std::string label_path {"/home/harun/Documents/neural_network/train-labels.idx1-ubyte"};
    
    MNIST test_set(data_path, label_path);
    for(int _{100}; _<105; ++_){
        test_set.print(_);
    }
}