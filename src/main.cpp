#include <cctype>
#include <iostream>
#include <neural_network.h>
#include <dataset.h>
#include <string>

void help(){
    std::cout << "\"Image idx > \"- Enter dataset image index to get prediction \nq - quit\nh - this help page\n";
}

int main(){
    std::cout << "Loading model...\n";
    NN nn("/home/harun/Documents/neural_network/model_91.nn");
    const std::string images_path {"/home/harun/Documents/neural_network/train-images.idx3-ubyte"};
    const std::string labels_path {"/home/harun/Documents/neural_network/train-labels.idx1-ubyte"};
    std::cout << "Loading dataset...\n";
    MNIST test_set(images_path, labels_path);

    std::string input{};
    std::cout << "--------------------------\n";
    std::cout << "\ntype h for help\n";
    while(true){
        std::cout << "Image idx > ";
        std::getline(std::cin, input);
        if(input == "q"){
            break;
        }
        if(input == "h"){
            help();
        }
        int idx = std::stoi(input);
        std::pair image = test_set.get_item(idx);
        test_set.print(idx); 
        std::cout << "Predicting...\n";
        nn.forward(image.second->t());
        Tensor<float> output = nn.get_output();
        int pred = output.max()[0];
        std::cout << "Prediction: " << pred << '\n';
    }


    return 0;
}