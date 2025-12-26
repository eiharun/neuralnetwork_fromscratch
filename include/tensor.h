#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <assert.h>

using std::vector;
using std::shared_ptr;

template<typename T>
class Tensor{
public:
    Tensor(T data); // 1x1 (scalar)
    Tensor(vector<T> data); // 1xM (array)
    Tensor(vector<T> data, vector<size_t> size); // NxM (matrix)
    Tensor(vector<vector<T>> data); // NxM (matrix)
    T& item(); // Scalar only
    T& at(size_t row, size_t col); // Aray/Matrix only
    T at(size_t row, size_t col) const; // Aray/Matrix only
    void get_size(size_t* row, size_t* col);
    size_t rows() const;
    size_t cols() const;
    vector<size_t> get_size() const;
    T dot(const Tensor& rhs) const;
    vector<size_t> max() const;
    Tensor<T> flatten() const;
    Tensor t() const; // transpose
    Tensor operator+(const Tensor& rhs) const;
    Tensor operator-(const Tensor& rhs) const;
    Tensor operator*(const Tensor& rhs) const;
    Tensor operator+(T scalar) const;
    Tensor operator-(T scalar) const;
    Tensor operator*(T scalar) const;
private:
    vector<size_t> m_size;
    vector<size_t> m_stride; //stride to next col,row; 5x3 = 3,1 stride = m_size[1],1
    vector<T> m_data; // ROW MAJOR
};

