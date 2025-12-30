#include "tensor.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ios>
#include <neural_network.h>
#include <random>
#include <iostream>
#include <fstream>

NN::NN(vector<int> nodes_per_layer, float learning_rate)
: m_learning_rate(learning_rate)
, m_target(vector<float>(nodes_per_layer[nodes_per_layer.size()-1],0)){
    m_deltas.resize(nodes_per_layer.size()-1, Tensor<float>(0));
    std::random_device rd;
    std::mt19937 engine(rd());
    int num_layers = nodes_per_layer.size();
    LIB_ASSERT(num_layers>1, "NN Must have more than 1 total layer");
    // Construct weights for each layer
    for(int i=0; i<num_layers-1; ++i){ // Disregard last output layer
        int num_nodes = nodes_per_layer[i];
        int num_nodes_next = nodes_per_layer[i+1];
        vector<vector<float>> weights(num_nodes_next, vector<float>(num_nodes));
        // col = num_nodes
        // row = num_nodes_next
        float scale = std::sqrt(2.0f/(num_nodes));
        std::normal_distribution<float> dist(0.0f, scale);
        for(int j=0; j<num_nodes_next; ++j){
            for(int k=0; k<num_nodes; ++k){
                weights[j][k] = dist(engine);
            }
        }
        m_weights.push_back(Tensor<float>(weights));
    }
    // Construct biases for each layer
    for(int i=1; i<num_layers; ++i){ // Disregard first input layer
        int num_nodes = nodes_per_layer[i];
        vector<vector<float>> biases(num_nodes, vector<float>(1));
        for(int j=0; j<num_nodes; ++j){
            biases[j][0] = 0;
        }
        m_biases.push_back(Tensor<float>(biases));
    }
    // Construct each layer Tensor
    for(int i=0; i<num_layers; ++i){ 
        int num_nodes = nodes_per_layer[i];
        vector<vector<float>> layers(num_nodes, vector<float>(1,0));
        m_original_layers.push_back(Tensor<float>(layers));
        m_activated_layers.push_back(Tensor<float>(layers));
    }

    std::cout << "\nWeights:\n";
    for(int i=0; i<m_weights.size(); ++i){
        std::cout << "m_weights[" << i << "]: (" << m_weights[i].rows() << "," << m_weights[i].cols() << ")\n";
    }
}

NN::NN(std::string filename, float learning_rate): NN(load_construct(filename), learning_rate){
    load_data(filename);
}

void NN::forward(Tensor<float>&& input){
    LIB_ASSERT(m_original_layers[0].rows() == input.rows(), "NN::forward Input row mismatch");
    LIB_ASSERT(m_original_layers[0].cols() == input.cols(), "NN::forward Input col mismatch");
    
    m_original_layers[0] = std::move(input);
    for(int i=0; i<m_original_layers.size()-1; ++i){
        m_original_layers[i+1] = (m_weights[i]*m_original_layers[i]) + m_biases[i];
        m_activated_layers[i+1] = m_original_layers[i+1];
        apply_activation(i+1);
    }
}

Tensor<float> NN::get_output(){
    return m_activated_layers[m_activated_layers.size()-1];
}

float NN::calculate_loss(Tensor<float> target){
    Tensor<float> output = m_activated_layers[m_activated_layers.size()-1];
    size_t r = output.rows();
    
    // float sum = 0.0f;
    // for(int i=0; i<r; ++i){
    //     float diff = output.at(i, 0) - target.at(i, 0);
    //     sum += diff * diff;
    // }
    // return sum / r;
        
    // Cross entropy loss
    float loss = 0.0f;
    for(int i=0; i<r; ++i){
        float y_true = target.at(i,0);
        float y_pred = std::max(output.at(i, 0), 1e-7f);  // Prevent log(0)
        loss += -y_true * std::log(y_pred);
    }
    return loss;
}

void NN::apply_activation(int layer){
    // Fast Sigmoid
    vector<size_t> size = m_activated_layers[layer].get_size();
    LIB_ASSERT(size[1] == 1, "Apply activation layer must be a column vector");
    if(layer == m_original_layers.size()-1){
        m_activated_layers[layer] = softmax(m_activated_layers[layer]);
    }
    else{
        for(int i=0; i<size[0]; ++i){
            m_activated_layers[layer].at(i,0) = LeakyReLU(m_activated_layers[layer].at(i,0));
        }
    }
}

Tensor<float> NN::softmax(const Tensor<float>& input){
    vector<size_t> size = input.get_size();
    vector<vector<float>> output(size[0], vector<float>(1));  // ← Column vector (n, 1)!
    vector<size_t> max_idx = input.max();
    float max_val = input.at(max_idx[0], max_idx[1]);
    double exp_sum{};
    for(size_t i{}; i<size[0]; ++i){
        output[i][0] = std::exp(input.at(i,0) - max_val); // Sub max_val to stabilize
        exp_sum += output[i][0];
    }
    for(size_t i{}; i<output.size(); ++i){
        output[i][0] /= exp_sum;
    }
    return Tensor<float>(output);
}

float NN::LeakyReLU(float input){
    return input > 0 ? input : 0.01f * input;
}

float NN::LeakyReLU_derivative(float input){
    return input > 0 ? 1.0f : 0.01f;
}

float NN::fast_sigmoid(float input){
    return (input/(1+std::abs(input)));
}

float NN::fast_sigmoid_derivative(float input){
    return (1/((1+std::abs(input))*(1+std::abs(input))));
}

float NN::gradient(float real, float target, int n){
    return (2.0/n)*(real-target);
}

float NN::output_neuron_delta(float post_activation_real, float target, float pre_activation_real, int n){
    m_out_delta = gradient(post_activation_real, target, n) * LeakyReLU_derivative(pre_activation_real);
    return m_out_delta;
}

void NN::compute_output_delta(){
    Tensor<float> output_layer = m_activated_layers[m_activated_layers.size()-1];
    vector<vector<float>> out_delta(output_layer.rows(), vector<float>(1));

    for(int i=0; i<output_layer.rows(); ++i){
        // float delta = output_neuron_delta(m_activated_layers[m_original_layers.size()-1].at(i,0), m_target.at(i,0), output_layer.at(i,0), output_layer.rows());
        // out_delta.push_back(delta);
        // With softmax
        out_delta[i][0] = (output_layer.at(i,0) - m_target.at(i,0));
    }
    m_deltas[m_original_layers.size()-2] = Tensor<float>(out_delta);
}

float NN::hidden_neuron_delta(float pre_activation_real, int layer_idx, int neuron_idx){
    LIB_ASSERT(layer_idx < m_original_layers.size(), "hidden_delta: layer idx must be smaller than index of output layer");
    Tensor<float> next_deltas = m_deltas[layer_idx];
    float weights_sum{};
    Tensor<float> cur_layer = m_original_layers[layer_idx];
    for(int i=0; i<next_deltas.rows(); ++i){
        weights_sum += m_weights[layer_idx].at(i,neuron_idx) * next_deltas.at(i,0);
    }
    return weights_sum * LeakyReLU_derivative(pre_activation_real);
}

void NN::compute_hidden_deltas(){
    const size_t last_layer_idx = m_original_layers.size()-2;
    const size_t first_layer_idx = 1;
    // std::cout << "Computing hidden deltas:\n";
    // std::cout << "last_layer_idx: " << last_layer_idx << "\n";
    // std::cout << "first_layer_idx: " << first_layer_idx << "\n";
    Tensor<float> cur_layer = m_original_layers[last_layer_idx];
    if(m_original_layers.size() == 2){
        return;
    }
    LIB_ASSERT(last_layer_idx>=first_layer_idx, "comp hidden delta: last layer idx must be gt or eq to first layer idx");
    for(int l=last_layer_idx; l>first_layer_idx-1; --l){
        if(l>=1){
            cur_layer = m_original_layers[l];
        }
        vector<vector<float>> h_delta(cur_layer.rows(), vector<float>(1));
        for(int i=0; i<cur_layer.rows(); ++i){
            h_delta[i][0] = (hidden_neuron_delta(cur_layer.at(i,0), l, i));
        }
        m_deltas[l-1] = Tensor<float>(h_delta);
    }
}

void NN::backward(Tensor<float>&& target){
    const int last_idx = m_original_layers.size()-1;
    LIB_ASSERT(m_original_layers[last_idx].rows() == target.rows(), "NN::backward target row mismatch");
    LIB_ASSERT(m_original_layers[last_idx].cols() == target.cols(), "NN::backward target col mismatch");

    //  std::cout << "\n=== Backward Pass Debug ===\n";
    // std::cout << "Network has " << m_original_layers.size() << " layers\n";
    // for(int i=0; i<m_original_layers.size(); ++i){
    //     std::cout << "Layer " << i << ": (" << m_original_layers[i].rows() << "," << m_original_layers[i].cols() << ")\n";
    // }

    m_target = std::move(target);
    compute_output_delta();
    compute_hidden_deltas();

    // std::cout << "\nDeltas before transpose:\n";
    // for(int i=0; i<m_deltas.size(); ++i){
    //     std::cout << "m_deltas[" << i << "]: (" << m_deltas[i].rows() << "," << m_deltas[i].cols() << ")\n";
    //     std::cout << "\t" << m_deltas[i].at(0,0) << " | " << m_deltas[i].at(0,1)<< '\n';
    // }

    for(int w=0; w<m_weights.size(); ++w){
        Tensor<float> dL_dw = m_deltas[w] * m_activated_layers[w].t(); // Transpose original_layers[w]
        // std::cout << "dl_dw: " << dL_dw.at(0,0) << " " << dL_dw.at(1,0) << '\n';
        m_weights[w] = m_weights[w] - (dL_dw*m_learning_rate);
    }
    for(int b=0; b<m_biases.size(); ++b){
        m_biases[b] = m_biases[b] - (m_deltas[b]*m_learning_rate);
    }
}

void NN::update_lr(float new_lr){
    m_learning_rate = new_lr;
}

void NN::save(std::string filename){
    // Save weights and biases
    // Format:
    /*
    * First 2 bytes = MAGIC_NUMBER
    * Next 2 bytes total num of layers
    * Next *num* 4 bytes nodes per layer (including input and output)
    * Next 4 bytes num of weight row
    * Next 4 bytes num of weight col
    * Raw weights in row major
    * Repeat until all weights
    * Next 4 bytes num of bias row
    * Next 4 bytes num of bias col
    * Raw bias in row major
    * Repeat until all biases
    */
    std::ofstream model_file(filename, std::ios::binary);
    if(!model_file.is_open()){
        std::cerr << "Error opening file: " << filename << '\n';
        return;
    }
    std::cout << "Writing file... \n Verification: \n";
    model_file.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(short));
    const uint16_t num_layers = static_cast<short>(m_original_layers.size());
    std::cout << "Writing num layers\n";
    std::cout << "Written: " << num_layers << '\n';
    model_file.write(reinterpret_cast<const char*>(&num_layers), sizeof(short));
    uint32_t num_nodes{};
    std::cout << "Writing num nodes p layer\n";
    for(int i{}; i<num_layers; ++i){
        num_nodes = m_original_layers[i].rows();
        std::cout << "Written: " << num_nodes << '\n';
        model_file.write(reinterpret_cast<const char*>(&num_nodes), sizeof(uint32_t));
    }
    
    uint32_t weight_row{};
    uint32_t weight_col{};
    std::cout << "Writing weights\n";
    for(int n{}; n<num_layers-1; ++n){
        weight_row = m_weights[n].rows();
        weight_col = m_weights[n].cols();
        std::cout << "Written weight row: " << weight_row << '\n';
        std::cout << "Written weight col: " << weight_col << '\n';
        model_file.write(reinterpret_cast<const char*>(&weight_row), sizeof(uint32_t));
        model_file.write(reinterpret_cast<const char*>(&weight_col), sizeof(uint32_t));
        for(int i{}; i<m_weights[n].rows(); ++i){
            for(int j{}; j<m_weights[n].cols(); ++j){
                float val = m_weights[n].at(i,j);
                model_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        std::cout << "Written weights matrix\n";

    }

    uint32_t bias_num{};
    std::cout << "Writing Biases\n";
    for(int n{1}; n<num_layers-1; ++n){
        bias_num = m_biases[n].rows();
        std::cout << "Written bias num: " << bias_num << '\n';
        model_file.write(reinterpret_cast<const char*>(&bias_num), sizeof(uint32_t));
        for(int i{}; i<m_biases[n].rows(); ++i){
            for(int j{}; j<m_biases[n].cols(); ++j){
                float val = m_biases[n].at(i,j);
                model_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        std::cout << "Written bias matrix\n";
    }

    model_file.close();
}


vector<int> NN::load_construct(std::string filename){
    // Format:
    /*
    * First 2 bytes = MAGIC_NUMBER
    * Next 2 bytes total num of layers
    * Next *num* 4 bytes nodes per layer (including input and output)
    * Next 4 bytes num of weight row
    * Next 4 bytes num of weight col
    * Raw weights in row major
    * Repeat until all weights
    * Next 4 bytes num of bias row
    * Next 4 bytes num of bias col
    * Raw bias in row major
    * Repeat until all biases
    */
    std::ifstream model_file(filename, std::ios::in | std::ios::binary);
    if(!model_file.is_open()){
        std::cerr << "Cannot open file\n";
        return {};
    }
    char magic_number_b[2];
    model_file.read(magic_number_b, 2);
    int magic_number{ (magic_number_b[1] << 8)  | magic_number_b[0] };
    
    if(magic_number != 0x6767){
        std::cerr << "Incorrect magic number\n";
        return {};
    }
    char num_layers_b[2];
    model_file.read(num_layers_b, 2);
    int num_layers{ ( num_layers_b[1] << 8 | num_layers_b[0]) };
    vector<int> nodes_per_layer(num_layers); 
    for(int i{}; i<num_layers; ++i){
        uint32_t num_nodes{};
        model_file.read(reinterpret_cast<char*>(&num_nodes), sizeof(uint32_t));
        nodes_per_layer[i] = num_nodes;
    }
    model_file.close();
    return nodes_per_layer;
}

void NN::load_data(std::string filename){
    // Format:
    /*
    * First 2 bytes = MAGIC_NUMBER
    * Next 2 bytes total num of layers
    * Next *num* 4 bytes nodes per layer (including input and output)
    * Next 4 bytes num of weight row
    * Next 4 bytes num of weight col
    * Raw weights in row major
    * Repeat until all weights
    * Next 4 bytes num of bias row
    * Next 4 bytes num of bias col
    * Raw bias in row major
    * Repeat until all biases
    */
    std::ifstream model_file(filename, std::ios::in | std::ios::binary);
    if(!model_file.is_open()){
        std::cerr << "Cannot open file\n";
        return;
    }
    char magic_number_b[2];
    model_file.read(magic_number_b, 2);
    int magic_number{ (magic_number_b[1] << 8)  | magic_number_b[0] };
    
    if(magic_number != 0x6767){
        std::cerr << "Incorrect magic number\n";
        return;
    }

    char num_layers_b[2];
    model_file.read(num_layers_b, 2);
    int num_layers{ ( num_layers_b[1] << 8 | num_layers_b[0]) };
    vector<int> nodes_per_layer(num_layers); 
    for(int i{}; i<num_layers; ++i){
        uint32_t num_nodes{};
        model_file.read(reinterpret_cast<char*>(&num_nodes), sizeof(uint32_t));
        nodes_per_layer[i] = num_nodes;
    }
    // Read weights
    vector<Tensor<float>> weights(num_layers-1, Tensor<float>(0.0f));
    for(int i{}; i<num_layers-1; ++i){
        // Row, Col
        uint32_t row{};
        uint32_t col{};
        model_file.read(reinterpret_cast<char*>(&row), sizeof(uint32_t));
        model_file.read(reinterpret_cast<char*>(&col), sizeof(uint32_t));
        //Data
        vector<float> data(row*col);
        for(int j{}; j<row*col; ++j){
            if(j == (row*col)-1){
                volatile int s{};
            }
            uint32_t val{};
            model_file.read(reinterpret_cast<char*>(&val), sizeof(uint32_t));
            data[j] = static_cast<float>(val);
        }
        weights[i] = Tensor<float>(data, vector<size_t>{row,col});
    }
    
    // Read biases
    vector<Tensor<float>> biases(num_layers-1, Tensor<float>(0.0f));
    for(int i{}; i<num_layers-1; ++i){
        // Row, Col
        uint32_t row{};
        uint32_t col{1}; // row vector
        model_file.read(reinterpret_cast<char*>(&row), sizeof(uint32_t));
        //Data
        vector<float> data(row);
        for(int j{}; j<row; ++j){
            uint32_t val{};
            model_file.read(reinterpret_cast<char*>(&val), sizeof(uint32_t));
            data[j] = val;
        }
        biases[i] = Tensor<float>(data, vector<size_t>{row,col});
    }

    model_file.close();
}