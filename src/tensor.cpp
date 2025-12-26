#include <tensor.h>


template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;


template<typename T>
Tensor<T>::Tensor(T data): m_data{data}, m_size{1,1}, m_stride{}{}

template<typename T>
Tensor<T>::Tensor(vector<T> data): m_data{data}, m_size{1, data.size()}, m_stride{data.size(),1}{}

template<typename T>
Tensor<T>::Tensor(vector<T> data, vector<size_t> size): m_data{data}, m_size{size}, m_stride{size[1],1}{}

template<typename T>
Tensor<T>::Tensor(vector<vector<T>> data): m_size{data.size(),data[0].size()}{
    // Invariant check
    for (const auto& row : data)
        LIB_ASSERT(row.size() == m_size[1], "ragged input matrix");

    m_stride = {m_size[1], 1};
    m_data.resize(m_size[0] * m_size[1]);
    // Store in m_data in Row Major order
    for(size_t i=0; i<m_size[0]; ++i){
        for(size_t j=0; j<m_size[1]; ++j){
            m_data[m_stride[0]*i + m_stride[1]*j] = data[i][j];
        }
    }
}

template<typename T>
T& Tensor<T>::item(){
    LIB_ASSERT((m_size[0] == 1 && m_size[1] == 1), "Item not (1x1) scalar ");
    LIB_ASSERT(m_data.size() == 1, "Item not (data size != 1) scalar");
    return m_data[0];
}

template<typename T>
T& Tensor<T>::at(size_t row, size_t col){
    LIB_ASSERT(row < m_size[0], "row out of range");
    LIB_ASSERT(col < m_size[1], "col out of range");
    return m_data[m_stride[0]*row + m_stride[1]*col];
}

template<typename T>
T Tensor<T>::at(size_t row, size_t col) const{
    LIB_ASSERT(row < m_size[0], "row out of range");
    LIB_ASSERT(col < m_size[1], "col out of range");
    return m_data[m_stride[0]*row + m_stride[1]*col];
}

template<typename T>
vector<size_t> Tensor<T>::get_size() const {
    return vector(m_size);
}

template<typename T>
void Tensor<T>::get_size(size_t* row, size_t* col){
    *row = m_size[0];
    *col = m_size[1];
}

template<typename T>
size_t Tensor<T>::rows() const {
    return m_size[0];
}

template<typename T>
size_t Tensor<T>::cols() const {
    return m_size[1];
}

template<typename T>
vector<size_t> Tensor<T>::max() const {
    // Get the index of the largest element
    T max_val = at(0,0);
    vector<size_t> max_idx{0,0};
    for(size_t i{}; i<rows(); ++i){
        for(size_t j{}; j<cols(); ++j){
            if(max_val < at(i,j)){
                max_val = at(i,j);
                max_idx = {i,j};
            }
        }
    }
    return max_idx;
}

template<typename T>
Tensor<T> Tensor<T>::t() const {
    vector<vector<T>> data_t(m_size[1], vector<T>(m_size[0]));
    for(int i{}; i<rows(); ++i){
        for(int j{}; j<cols(); ++j){
            data_t[j][i] = at(i,j);
        }
    }
    return Tensor<T>(data_t);
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& rhs) const {
    vector<T> result;
    vector<size_t> size;
    if(rhs.m_size[0] == 1 && rhs.m_size[1] == 1){
        LIB_ASSERT(rhs.m_data.size() == 1, "RHS not scalar");
        result.reserve(m_data.size());
        for(T element: m_data){
            result.push_back(element + rhs.m_data[0]);
        }
        size = m_size;
    }
    else if(m_size[0] == 1 && m_size[1] == 1){
        LIB_ASSERT(m_data.size() == 1, "LHS not scalar");
        result.reserve(rhs.m_data.size());
        for(T element: rhs.m_data){
            result.push_back(element + m_data[0]);
        }
        size = rhs.m_size;
    }
    else{
        LIB_ASSERT(m_size[0] == rhs.m_size[0], "Tensor addition, row mismatch");
        LIB_ASSERT(m_size[1] == rhs.m_size[1], "Tensor addition, column mismatch");
        result.reserve(m_data.size());
        for(size_t i=0; i<m_data.size(); ++i){
            result.push_back(m_data[i] + rhs.m_data[i]);
        }
        size = m_size;
    }
    return Tensor(result, size);
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& rhs) const {
    vector<T> result;
    vector<size_t> size;
    if(rhs.m_size[0] == 1 && rhs.m_size[1] == 1){
        LIB_ASSERT(rhs.m_data.size() == 1, "RHS not scalar");
        result.reserve(m_data.size());
        for(T element: m_data){
            result.push_back(element - rhs.m_data[0]);
        }
        size = m_size;

    }
    else if(m_size[0] == 1 && m_size[1] == 1){
        LIB_ASSERT(m_data.size() == 1, "LHS not scalar");
        result.reserve(rhs.m_data.size());
        for(T element: rhs.m_data){
            result.push_back(element - m_data[0]);
        }
        size = rhs.m_size;
    }
    else{
        LIB_ASSERT(m_size[0] == rhs.m_size[0], "Tensor addition, row mismatch");
        LIB_ASSERT(m_size[1] == rhs.m_size[1], "Tensor addition, column mismatch");
        result.reserve(m_data.size());
        for(size_t i=0; i<m_data.size(); ++i){
            result.push_back(m_data[i] - rhs.m_data[i]);
        }
        size = m_size;
    }
    return Tensor(result, size);
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& rhs) const {
    LIB_ASSERT(m_size[1] == rhs.m_size[0], "Tensor multiplication, size mismatch {lhs col != rhs row}");
    vector<size_t> result_size = {m_size[0], rhs.m_size[1]};
    vector<vector<T>> result(result_size[0], vector<T>(result_size[1], 0));
    
    for (size_t i = 0; i < result_size[0]; ++i) {
        for (size_t k = 0; k < m_size[1]; ++k) {
            T a_ik = at(i, k);
            for (size_t j = 0; j < result_size[1]; ++j) {
                result[i][j] += a_ik * rhs.at(k, j);
            }
        }
    }
    return Tensor(result);
}

template<typename T>
Tensor<T> Tensor<T>::operator+(T scalar) const{
    vector<T> result;
    result.reserve(m_data.size());
    for(T element: m_data){
        result.push_back(element + scalar);
    }
    return Tensor(result, m_size);
}

template<typename T>
Tensor<T> Tensor<T>::operator-(T scalar) const{
    vector<T> result;
    result.reserve(m_data.size());
    for(T element: m_data){
        result.push_back(element - scalar);
    }
    return Tensor(result, m_size);
}

template<typename T>
Tensor<T> Tensor<T>::operator*(T scalar) const{
    vector<T> result;
    result.reserve(m_data.size());
    for(T element: m_data){
        result.push_back(element * scalar);
    }
    return Tensor(result, m_size);
}

