#include <tensor.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

TEST(TensorConstruct, Scalar){
    Tensor ti(6); // Int
    Tensor tf(3.14f); // Float
    Tensor td(5.121); // Double

    EXPECT_EQ(ti.item(), 6);
    EXPECT_FLOAT_EQ(tf.item(), 3.14f);
    EXPECT_DOUBLE_EQ(td.item(), 5.121);
}

TEST(TensorConstruct, OneDArray){
    const int SIZE {50};
    std::vector<int> ti_d(SIZE);
    std::vector<float> tf_d(SIZE);
    std::vector<double> td_d(SIZE);
    for(int i=0; i<SIZE; ++i){
        ti_d[i] = i+1;
        tf_d[i] = i/1.8f;
        td_d[i] = i/3.124;
    }
    Tensor ti(ti_d);
    Tensor tf(tf_d);
    Tensor td(td_d);
    std::vector<size_t> size_i = ti.get_size();
    std::vector<size_t> size_f = tf.get_size();
    std::vector<size_t> size_d = td.get_size();
    ASSERT_EQ(size_i[0], 1); // Rows
    ASSERT_EQ(size_i[1], SIZE); // Cols
    ASSERT_EQ(size_f[0], 1); // Rows
    ASSERT_EQ(size_f[1], SIZE); // Cols
    ASSERT_EQ(size_d[0], 1); // Rows
    ASSERT_EQ(size_d[1], SIZE); // Cols
    
    for(int i=0; i<SIZE; ++i){
        EXPECT_EQ(ti.at(0,i), i+1);
        EXPECT_EQ(tf.at(0,i), i/1.8f);
        EXPECT_EQ(td.at(0,i), i/3.124);
    }
}

TEST(TensorConstruct, NxMMatrix){
    const int N {50};
    const int M {8}; 
    std::vector<std::vector<int>> ti_d(N, std::vector<int>(M));
    std::vector<std::vector<float>> tf_d(N, std::vector<float>(M));
    std::vector<std::vector<double>> td_d(N, std::vector<double>(M));
    for(int i=0; i<N; ++i){
        for(int j=0; j<M; ++j){
            ti_d[i][j] = (i+1)+(j*2);
            tf_d[i][j] = (i/1.8f)-j;
            td_d[i][j] = (i/3.124)*(1.0/(1+j));
        }
    }
    Tensor ti(ti_d);
    Tensor tf(tf_d);
    Tensor td(td_d);
    std::vector<size_t> size_i = ti.get_size();
    std::vector<size_t> size_f = tf.get_size();
    std::vector<size_t> size_d = td.get_size();
    ASSERT_EQ(size_i[0], N); // Rows
    ASSERT_EQ(size_i[1], M); // Cols
    ASSERT_EQ(size_f[0], N); // Rows
    ASSERT_EQ(size_f[1], M); // Cols
    ASSERT_EQ(size_d[0], N); // Rows
    ASSERT_EQ(size_d[1], M); // Cols
    for(int i=0; i<N; ++i){
        for(int j=0; j<M; ++j){
            EXPECT_EQ(ti.at(i, j), (i+1)+(j*2));
            EXPECT_EQ(tf.at(i, j), (i/1.8f)-j);
            EXPECT_EQ(td.at(i, j), (i/3.124)*(1.0/(1+j)));
        }
    }
}

TEST(TensorOperations, ScalarF){
    const int N {10};
    const int M {20}; 
    Tensor scalar(5.0f);
    std::vector<float> td_f(N, 1);
    std::vector<std::vector<float>> t2d_f(N, std::vector<float>(M, 2.1));
    
    Tensor t_f(td_f);
    Tensor r1 = t_f + scalar;
    Tensor r1_1 = t_f + 5;
    for(int i=0; i<N; ++i){
        EXPECT_FLOAT_EQ(r1.at(0,i), 6.0f);
        EXPECT_FLOAT_EQ(r1.at(0,i), r1_1.at(0,i));
    }
    Tensor t2_f(t2d_f);
    Tensor r2 = t2_f + scalar;
    Tensor r2_1 = t2_f + 5.0;
    for(int i=0; i<N; ++i){
        for(int j=0; j<M; ++j){
            EXPECT_FLOAT_EQ(r2.at(i,j), 2.1f + 5.0f);
            EXPECT_FLOAT_EQ(r2.at(i,j), r2_1.at(i,j));
        }
    }
}

TEST(TensorOperations, OneDArray){
    
}

TEST(TensorOperations, NxMMatrix){
    
}

TEST(TensorOperations, Transpose){
    const int N {10};
    const int M {20};
    std::vector<std::vector<float>> data(N, std::vector<float>(M));
    for(int i{}; i<N; ++i){
        for(int j{}; j<M; ++j){
            data[i][j] = i+j;
        }
    }
    Tensor t(data);
    EXPECT_EQ(t.rows(), N);
    EXPECT_EQ(t.cols(), M);
    Tensor t_t = t.t();
    EXPECT_EQ(t_t.rows(), M);
    EXPECT_EQ(t_t.cols(), N);
    
}

