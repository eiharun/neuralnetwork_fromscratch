#pragma once
#include <tensor.h>
#include <assert.h>

class NN{
public:
    NN(vector<int> nodes_per_layer, float learning_rate);
    NN(std::string filename, float learning_rate=0.01f); // Load saved model
    void forward(Tensor<float>&& input);
    float calculate_loss(Tensor<float> truth);
    Tensor<float> get_output();
    // void backward();
    void backward(Tensor<float>&& target);
    void update_lr(float new_lr);
    void save(std::string filename);
private:
    vector<int> load_construct(std::string filename);
    void load_data(std::string filename);
    const short MAGIC_NUMBER {0x6767};
    void apply_activation(int layer);
    float LeakyReLU(float input);
    float LeakyReLU_derivative(float input);
    Tensor<float> softmax(const Tensor<float>& input);
    float fast_sigmoid(float input);
    float fast_sigmoid_derivative(float input);
    float gradient(float real, float target, int n);
    float output_neuron_delta(float real, float target, float pre_activation_real, int n);
    void compute_output_delta();
    float hidden_neuron_delta(float pre_activation_real, int layer_idx, int neuron_idx);
    void compute_hidden_deltas();

    Tensor<float> m_target;
    
    float m_learning_rate{0.01};
    float m_out_delta;
    vector<Tensor<float>> m_original_layers;
    vector<Tensor<float>> m_activated_layers;
    vector<Tensor<float>> m_deltas; 
    vector<Tensor<float>> m_weights;    
    vector<Tensor<float>> m_biases;    
};