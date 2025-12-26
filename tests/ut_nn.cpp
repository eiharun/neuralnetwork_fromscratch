#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <neural_network.h>
#include <string>
#include <tensor.h>
#include <gtest/gtest.h>
#include <vector>
#include <dataset.h>

TEST(TestNN, Initial){
    const int input_size = 3;
    std::vector<float> input_data(input_size,0);
    input_data[1] = 255;
    input_data[2] = 200;
    Tensor<float> input(input_data);
    std::vector<int> nodes_p_layer(3);
    nodes_p_layer[0] = input_size;
    nodes_p_layer[1] = 2;
    nodes_p_layer[2] = 1;
    NN nn(nodes_p_layer, 0.8);
    nn.forward(std::move(input.t()));
}

TEST(Perceptron, Initial){
    // float truth_f {1};
    GTEST_SKIP();
    std::vector<float> truth_v {1.0};
    Tensor<float> truth = Tensor<float>(truth_v);
    const int input_size = 2;
    std::vector<float> input_data(input_size);
    input_data[0] = 150;
    input_data[1] = 0.9;
    Tensor<float> input = Tensor<float>(input_data);
    std::vector<int> nodes_p_layer(2);
    nodes_p_layer[0] = input_size;
    nodes_p_layer[1] = 1;
    NN nn(nodes_p_layer, 0.8);
    std::shared_ptr<Tensor<float>> output;
    nn.forward(std::move(input.t()));
    // std::cout << output->item() input<< '\n';
    // float loss = nn.calculate_loss(truth);
    // std::cout << loss << '\n';
    // Gradient = pred - truth
    nn.backward(std::move(truth));
}

#include <random>
TEST(MNIST, Initial){
    const std::string data_path {"/home/harun/Documents/neural_network/train-images.idx3-ubyte"};
    const std::string label_path {"/home/harun/Documents/neural_network/train-labels.idx1-ubyte"};
    MNIST train_set(data_path, label_path);
    std::vector<int> nodes_p_layer{28*28, 28*28, 10};
    float LR {0.002};
    NN nn(nodes_p_layer, LR);
    // Start for loop train images * epoch
    std::vector<int> idxs(train_set.get_length());
    std::generate(idxs.begin(), idxs.end(), [n=0]() mutable{return n++;});
    std::random_device rd;
    std::mt19937 engine(rd());
    const int epoch = 10;
    const float lr_max = 0.01;
    const float lr_min = 0.0001;

    const std::string test_data_path {"/home/harun/Documents/neural_network/t10k-images.idx3-ubyte"};
    const std::string test_label_path {"/home/harun/Documents/neural_network/t10k-labels.idx1-ubyte"};
    MNIST test_set(test_data_path, test_label_path);
    float last_accuracy{};
    for(int _{}; _<epoch; ++_){
        float avg_loss{};
        std::shuffle(idxs.begin(),idxs.end(), engine);
        std::cout << "\nEpoch: " << _+1 << '\n';
        
        if(_ >= 3){ // Decay learning rate
            float new_lr = LR * std::pow(0.9, _-3);
            nn.update_lr(new_lr);
            LR = new_lr;
        }

        std::cout << "LR: " << LR << '\n';
        
        auto start = std::chrono::steady_clock::now();
        auto iter_start = std::chrono::steady_clock::now();
        double total_dur{};
        for(int i{}; i<train_set.get_length(); ++i){
            iter_start = std::chrono::steady_clock::now();
            std::cout << "\rImg: " << i+1;
            std::pair<int, shared_ptr<Tensor<float>>> pair = train_set.get_item(idxs[i]);
            nn.forward(std::move(pair.second->t()));
            //Construct target tensor
            vector<float> target_data(10, 0.0f);
            target_data[pair.first] = 1.0f;
            Tensor<float> target(target_data);
            nn.backward(std::move(target.t()));
            if((i+1)%1000 == 0){
                float loss = nn.calculate_loss(target.t());
                // std::cout << " Loss :" << loss << '\n';
                avg_loss += loss;
            }
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - iter_start);
            total_dur += duration.count();
        }
        std::cout << "\rAvg Image Duration: " << total_dur/train_set.get_length() << " ms\n";
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
        std::cout << "Duration: " << duration.count() << " seconds\n";
        std::cout << " Average Epoch Loss: " << avg_loss/60 << '\n';
        // Test accuracy of epoch
        float accuracy{0};
        for(int i{}; i<test_set.get_length(); ++i){
            std::pair<int, shared_ptr<Tensor<float>>> pair = test_set.get_item(i);
            nn.forward(std::move(pair.second->t()));
            Tensor<float> output = nn.get_output();
            if(output.max()[0] == pair.first){
                accuracy += 1;
            }
        }

        float num_correct = accuracy;    
        accuracy /= test_set.get_length();
        if(accuracy > last_accuracy){
            last_accuracy = accuracy;
        }
        std::cout << "Epoch Accuracy: " << accuracy*100 << "%\n";

    }
    std::cout << "\n==========Training Complete ==========\n";
    int accuracy = static_cast<int>(last_accuracy);
    std::string out_file {"/home/harun/Documents/neural_network/model_"+std::to_string(accuracy)+".nn"};
    nn.save(out_file);
    // std::cout << "\n==========Model File Saved==========\n";
    
}