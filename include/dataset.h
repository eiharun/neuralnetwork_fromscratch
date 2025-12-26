#include <tensor.h>
#include <string>

class Dataset {
public:
    virtual std::pair<int, std::shared_ptr<Tensor<float>>> get_item(int index) = 0;
    virtual int get_length() = 0;
};

class MNIST : public Dataset {
public:
    MNIST(std::string filepath, std::string labels_path);
    std::pair<int, std::shared_ptr<Tensor<float>>> get_item(int index) override;
    int get_length() override;
    void print(int idx);
private:
    void load_images(std::string data_path);
    void load_labels(std::string labels_path);
    enum class classes {ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE};
    enum class type{ NONE=0, U_BYTE=0x08, S_BYTE, SHORT=0x0B, INT, FLOAT, DOUBLE};
    vector<size_t> m_dims;
    bool m_file_loaded{false};
    MNIST::type m_type{};
    vector<shared_ptr<Tensor<float>>> m_data;
    vector<int> m_labels;
};