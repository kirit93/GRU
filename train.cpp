#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cmath>
#include <math.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>

using namespace Eigen;

void write_binary_matrix(std::string filename, const MatrixXf& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename MatrixXf::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename MatrixXf::Index));
    out.write((char*) (&cols), sizeof(typename MatrixXf::Index));
    out.write((char*) matrix.data(), rows * cols * sizeof(typename MatrixXf::Scalar) );
    out.close();
}

void read_binary_matrix(std::string filename, MatrixXf& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename MatrixXf::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename MatrixXf::Index));
    in.read((char*) (&cols),sizeof(typename MatrixXf::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows * cols * sizeof(typename MatrixXf::Scalar) );
    in.close();
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_grad(float x) {
    return (1 - x) * x;
}

float relu(float x) {
    if(x <= 0)
        return 0.001 * x;
    else
        return x;
}

float relu_grad(float x) {
   if(x <= 0)
        return 0.001;
    else
        return 1;
}

/* Tanh activation function */
float tanh_activation(float x) {
    return tanh(x);
}

/* Tanh of sigmoid function */
float tanh_grad(float x) {
    return 1 - (x * x);
}

float log_matrix(float x) {
    if(x <= 0)
        return 0.0;
    else
        return log(x);
}

float max = 0;
float softmax(float x) {
    return exp(x - max);
}

MatrixXf loss_softmax_grad(MatrixXf& y, MatrixXf& o) {
    return o - y;
}

void dropout(MatrixXf& m, float p) {
    int limit = m.rows() * m.cols() * p;
    int r, c;
    for (int i = 0; i < limit; ++i)
    {
        srand(clock());
        r = rand() % m.rows();
        c = rand() % m.cols();
        m(r, c) = 0;
    }
}

float div_temp(float x) {
    float temp = 0.5;
    if(x <= 0)
        x = 0;
    else
        x = log(x) / temp;
    return x;
}

void forward_propagation(MatrixXf& U_z, MatrixXf& U_r, MatrixXf& U_h, MatrixXf& W_z, MatrixXf& W_r, MatrixXf& W_h, MatrixXf& V, MatrixXf& X, MatrixXf& Y, MatrixXf& O, MatrixXf& S, MatrixXf& E, MatrixXf& z, MatrixXf& r, MatrixXf& h, int time_steps, int input_dim, int hidden_dim, int output_dim) {
    /* Forward propagation Step - returns z, r, h, S, O, E */
    /*
    * z_t = ReLU[(X_t * U_z) + (S_t-1 * W_z)]
    * r_t = ReLU[(X_t * U_r) + (S_t-1 * W_r)]
    * h_t = ReLU[(X_t * U_h) + ((S_t-1 o r_t) * W_h)]
    * S_t = tanh[((1 - z_t) o h_t) + (z_t o S_t-1)]
    * O = softmax(S_t * V)
    * E = - 1/N sum(Y o log(O))
    * where * is matrix multiplication and o is componentwise multiplication
    */

    MatrixXf temp           = MatrixXf::Zero(1, hidden_dim);
    MatrixXf temp_output    = MatrixXf::Zero(1, output_dim);
    MatrixXf temp_hidden    = MatrixXf::Zero(1, hidden_dim);
    for(int i = 0; i < time_steps; i++) {

        dropout(S, 0.3);

        temp            = (X.row(i) * (U_z)) + (S.row(i) * (W_z));
        temp.eval();
        z.row(i)        = temp.unaryExpr(&sigmoid);
        z.eval();
        
        // std::cout << "Z-" << i << std::endl;
        // std::cout << z << std::endl;

        temp            = (X.row(i) * (U_r)) + (S.row(i) * (W_r));
        temp.eval();
        r.row(i)        = temp.unaryExpr(&sigmoid);
        r.eval();

        // std::cout << "R-" << i << std::endl;
        // std::cout << r << std::endl;

        temp            = (X.row(i) * (U_h)) + (S.row(i).cwiseProduct(r.row(i))) * (W_h);
        temp.eval();
        h.row(i)        = temp.unaryExpr(&tanh_activation);
        h.eval();

        // std::cout << "H-" << i << std::endl;
        // std::cout << h << std::endl;

        temp_hidden     = (MatrixXf::Ones(1, hidden_dim) - z.row(i)).cwiseProduct(h.row(i)) + z.row(i).cwiseProduct(S.row(i));
        temp_hidden.eval();
        S.row(i + 1)    = temp_hidden;
        S.eval();

        // std::cout << "S-" << i << std::endl;
        // std::cout << S << std::endl;

        temp_output     = S.row(i + 1) * (V);
        temp_output.eval();

        max             = temp_output.maxCoeff();
        temp_output     = temp_output.unaryExpr(&softmax);
        temp_output.eval();

        O.row(i)        = temp_output / temp_output.sum();
        O.eval();

        // std::cout << "O-" << i << std::endl;
        // std::cout << O << std::endl;

        temp_output     = O.row(i);
        temp_output.eval();
    }

    E(0, 0)         = -1 * (Y.row(0).cwiseProduct(temp_output.unaryExpr(&log_matrix)).sum());
    E.eval();

}

float calculate_cost(MatrixXf& E, int time_steps) {
    /* Possibly - Move to forward propagation or Train */
    return E.sum();//(E.sum() / time_steps);
}

void back_propagation(MatrixXf& V, MatrixXf& U_z, MatrixXf& U_r, MatrixXf& U_h, MatrixXf& W_z, MatrixXf& W_r, MatrixXf& W_h, MatrixXf& dU_z, MatrixXf& dU_r, MatrixXf& dU_h, MatrixXf& dW_z, MatrixXf& dW_r, MatrixXf& dW_h, MatrixXf& dV, MatrixXf& z, MatrixXf& r, MatrixXf& h, MatrixXf& O, MatrixXf& S, MatrixXf& E, MatrixXf& X, MatrixXf& Y, int input_dim, int hidden_dim, int output_dim, int time_steps) {
    /* gradients = dLdV, dLdU0, dLdU1, dLdU2, dLdW0, dLdW1, dLdW2 */
    int time_step;
    dU_z = MatrixXf::Zero(input_dim, hidden_dim);
    dU_r = MatrixXf::Zero(input_dim, hidden_dim);
    dU_h = MatrixXf::Zero(input_dim, hidden_dim);
    dW_z = MatrixXf::Zero(hidden_dim, hidden_dim);
    dW_r = MatrixXf::Zero(hidden_dim, hidden_dim);
    dW_h = MatrixXf::Zero(hidden_dim, hidden_dim);

    MatrixXf ds_0, dsr, ds_single, ds_cur, ds_cur_bk, dz, delta_y, db_V, dreluInput_z, dreluInput_r, dreluInput_h, temp_r, temp_z, temp_h, temp_S, temp_X, temp_W, temp_U, temp_V;
    dsr       = MatrixXf::Zero(1, hidden_dim);
    temp_S    = MatrixXf::Zero(1, hidden_dim);
    temp_z    = MatrixXf::Zero(1, hidden_dim);
    temp_r    = MatrixXf::Zero(1, hidden_dim);
    temp_h    = MatrixXf::Zero(1, hidden_dim);
    temp_X    = MatrixXf::Zero(1, input_dim);
    ds_cur_bk = MatrixXf::Zero(1, hidden_dim);

    delta_y   = O.row(time_steps - 1) - Y.row(0);


    dV = MatrixXf::Zero(hidden_dim, output_dim);
    temp_S = S.row(time_steps);
    dV = temp_S.transpose().eval() * delta_y;
    ds_single = delta_y * V.transpose().eval();

    ds_cur = ds_single;

    for(time_step = time_steps - 1; time_step >= 0; time_step--) {
        ds_cur_bk = ds_cur;
        temp_S = S.row(time_step);
        temp_r = r.row(time_step);
        temp_z = z.row(time_step);
        temp_h = h.row(time_step);
        temp_X = X.row(time_step);

        dreluInput_h = ds_cur.cwiseProduct(MatrixXf::Ones(1, hidden_dim) - temp_z).cwiseProduct(temp_h.unaryExpr(&tanh_grad));//.cwiseProduct(temp_S.unaryExpr(&tanh_grad));
        temp_U = (temp_X.transpose().eval() * dreluInput_h);
        dU_h = dU_h + temp_U;

        temp_W = ((temp_S.cwiseProduct(temp_r)).transpose().eval() * dreluInput_h);
        dW_h = dW_h + temp_W;

        dsr = dreluInput_h * W_h.transpose().eval();
        ds_cur = dsr.cwiseProduct(temp_r);
        dreluInput_r = dsr.cwiseProduct(temp_S).cwiseProduct(temp_r.unaryExpr(&sigmoid_grad));

        temp_U = (temp_X.transpose().eval() * dreluInput_r);
        dU_r = dU_r + temp_U;

        temp_W = (temp_S.transpose().eval() * dreluInput_r);
        dW_r = dW_r + temp_W;

        ds_cur = ds_cur + (dreluInput_r * W_r.transpose().eval());
        ds_cur = ds_cur + ds_cur_bk.cwiseProduct(temp_z);
        dz = ds_cur_bk.cwiseProduct(temp_S - temp_h);
        dreluInput_z = dz.cwiseProduct(temp_z.unaryExpr(&sigmoid_grad));

        temp_U = (temp_X.transpose().eval() * dreluInput_z);
        dU_z = dU_z + temp_U;

        temp_W = (temp_S.transpose().eval() * dreluInput_z);
        dW_z = dW_z + temp_W;

        ds_cur = ds_cur + (dreluInput_z * W_z.transpose().eval());
    }

    dU_z /= time_steps;
    dU_r /= time_steps;
    dU_h /= time_steps;
    dW_z /= time_steps;
    dW_r /= time_steps;
    dW_h /= time_steps;
    dV   /= time_steps;

    dU_z.eval();
    dU_r.eval();
    dU_h.eval();
    dW_z.eval();
    dW_r.eval();
    dW_h.eval();
    dV.eval();
}

void divide_matrix(MatrixXf& gradient_total, MatrixXf gradient, MatrixXf cache) {

    for (int i = 0; i < gradient_total.rows(); ++i)
    {
        for (int j = 0; j < gradient_total.cols(); ++j)
        {
            gradient_total(i, j) = ( gradient(i, j) / ( sqrt(cache(i, j)) + 0.00000001) );
        }
    }
}

void rms_prop(MatrixXf& U_z, MatrixXf& U_r, MatrixXf& U_h, MatrixXf& W_z, MatrixXf& W_r, MatrixXf& W_h, MatrixXf& V, MatrixXf& U_z_grad, MatrixXf& U_r_grad, MatrixXf& U_h_grad, MatrixXf& W_z_grad, MatrixXf& W_r_grad, MatrixXf& W_h_grad, MatrixXf& V_grad, MatrixXf& cache_U_z, MatrixXf& cache_U_r, MatrixXf& cache_U_h, MatrixXf& cache_W_z, MatrixXf& cache_W_r, MatrixXf& cache_W_h, MatrixXf& cache_V, float learning_rate, int input_dim, int hidden_dim, int output_dim) {

    float decay = 0.9;

    MatrixXf U_z_grad_total = MatrixXf::Zero(input_dim, hidden_dim);
    MatrixXf U_r_grad_total = MatrixXf::Zero(input_dim, hidden_dim);
    MatrixXf U_h_grad_total = MatrixXf::Zero(input_dim, hidden_dim);
    MatrixXf W_z_grad_total = MatrixXf::Zero(hidden_dim, hidden_dim);
    MatrixXf W_r_grad_total = MatrixXf::Zero(hidden_dim, hidden_dim);
    MatrixXf W_h_grad_total = MatrixXf::Zero(hidden_dim, hidden_dim);
    MatrixXf V_grad_total   = MatrixXf::Zero(hidden_dim, output_dim);

    cache_U_z = decay * cache_U_z + (1 - decay) * (U_z_grad.cwiseProduct(U_z_grad)).eval();
    cache_U_r = decay * cache_U_r + (1 - decay) * (U_r_grad.cwiseProduct(U_r_grad)).eval();
    cache_U_h = decay * cache_U_h + (1 - decay) * (U_h_grad.cwiseProduct(U_h_grad)).eval();
    cache_W_z = decay * cache_W_z + (1 - decay) * (W_z_grad.cwiseProduct(W_z_grad)).eval();
    cache_W_r = decay * cache_W_r + (1 - decay) * (W_r_grad.cwiseProduct(W_r_grad)).eval();
    cache_W_h = decay * cache_W_h + (1 - decay) * (W_h_grad.cwiseProduct(W_h_grad)).eval();
    cache_V   = decay * cache_V   + (1 - decay) * (V_grad.cwiseProduct(V_grad)).eval();

    cache_U_z.eval();
    cache_U_r.eval();
    cache_U_h.eval();
    cache_W_z.eval();
    cache_W_r.eval();
    cache_W_h.eval();
    cache_V.eval();

    divide_matrix(U_z_grad_total, U_z_grad, cache_U_z);
    divide_matrix(U_r_grad_total, U_r_grad, cache_U_r);
    divide_matrix(U_h_grad_total, U_h_grad, cache_U_h);
    divide_matrix(W_z_grad_total, W_z_grad, cache_W_z);
    divide_matrix(W_r_grad_total, W_r_grad, cache_W_r);
    divide_matrix(W_h_grad_total, W_h_grad, cache_W_h);
    divide_matrix(V_grad_total, V_grad, cache_V);

    U_z_grad_total.eval();
    U_r_grad_total.eval();
    U_h_grad_total.eval();
    W_z_grad_total.eval();
    W_r_grad_total.eval();
    W_h_grad_total.eval();
    V_grad_total.eval();

    U_z -= learning_rate * U_z_grad_total;
    U_r -= learning_rate * U_r_grad_total;
    U_h -= learning_rate * U_h_grad_total;
    W_z -= learning_rate * W_z_grad_total;
    W_r -= learning_rate * W_r_grad_total;
    W_h -= learning_rate * W_h_grad_total;
    V   -= learning_rate * V_grad_total;

    U_z_grad  = MatrixXf::Zero(input_dim, hidden_dim);
    U_r_grad  = MatrixXf::Zero(input_dim, hidden_dim);
    U_h_grad  = MatrixXf::Zero(input_dim, hidden_dim);
    W_z_grad  = MatrixXf::Zero(hidden_dim, hidden_dim);
    W_r_grad  = MatrixXf::Zero(hidden_dim, hidden_dim);
    W_h_grad  = MatrixXf::Zero(hidden_dim, hidden_dim);
    V_grad    = MatrixXf::Zero(hidden_dim, output_dim);

    U_z_grad.eval();
    U_r_grad.eval();
    U_h_grad.eval();
    W_z_grad.eval();
    W_r_grad.eval();
    W_h_grad.eval();
    V_grad.eval();

    U_z.eval();
    U_r.eval();
    U_h.eval();
    W_z.eval();
    W_r.eval();
    W_h.eval();
    V.eval();
}


void gradient_descent(MatrixXf& U_z, MatrixXf& U_r, MatrixXf& U_h, MatrixXf& W_z, MatrixXf& W_r, MatrixXf& W_h, MatrixXf& V,
            MatrixXf& U_z_grad, MatrixXf& U_r_grad, MatrixXf& U_h_grad, MatrixXf& W_z_grad, MatrixXf& W_r_grad, MatrixXf& W_h_grad, MatrixXf& V_grad,
            float learning_rate, int input_dim, int hidden_dim, int output_dim) {

    V   -= learning_rate * V_grad;
    U_z -= learning_rate * U_z_grad;
    U_r -= learning_rate * U_r_grad;
    U_h -= learning_rate * U_h_grad;
    W_z -= learning_rate * W_z_grad;
    W_r -= learning_rate * W_r_grad;
    W_h -= learning_rate * W_h_grad;

    U_z.eval();
    U_r.eval();
    U_h.eval();
    W_z.eval();
    W_r.eval();
    W_h.eval();
    V.eval();

    U_z_grad  = MatrixXf::Zero(input_dim, hidden_dim);
    U_r_grad  = MatrixXf::Zero(input_dim, hidden_dim);
    U_h_grad  = MatrixXf::Zero(input_dim, hidden_dim);
    W_z_grad  = MatrixXf::Zero(hidden_dim, hidden_dim);
    W_r_grad  = MatrixXf::Zero(hidden_dim, hidden_dim);
    W_h_grad  = MatrixXf::Zero(hidden_dim, hidden_dim);
    V_grad    = MatrixXf::Zero(hidden_dim, output_dim);

    U_z_grad.eval();
    U_r_grad.eval();
    U_h_grad.eval();
    W_z_grad.eval();
    W_r_grad.eval();
    W_h_grad.eval();
    V_grad.eval();
}

void init_matrix(MatrixXf& X, float dimension_row, float dimension_col) {
    float upperlimit = 1.0 * sqrt(1.0 / (float)dimension_row);
    float lowerlimit = -1.0 * sqrt(1.0 / (float)dimension_row);;
    float range = upperlimit - lowerlimit;

    srand(clock());
    X = MatrixXf::Random(dimension_row, dimension_col);
    X = (X + MatrixXf::Constant(dimension_row, dimension_col, 1.0)) * (range / 2.0);
    X = (X + MatrixXf::Constant(dimension_row, dimension_col, lowerlimit));
}

void init_weight_matrices(MatrixXf& U_z, MatrixXf& U_r, MatrixXf& U_h, MatrixXf& W_z, MatrixXf& W_r, MatrixXf& W_h, MatrixXf& V, int input_dim, int output_dim, int hidden_dim) {
    init_matrix(U_z, input_dim, hidden_dim);
    init_matrix(U_r, input_dim, hidden_dim);
    init_matrix(U_h, input_dim, hidden_dim);
    init_matrix(W_z, hidden_dim, hidden_dim);
    init_matrix(W_r, hidden_dim, hidden_dim);
    init_matrix(W_h, hidden_dim, hidden_dim);
    init_matrix(V, hidden_dim, output_dim);

    U_z.eval();
    U_r.eval();
    U_h.eval();
    W_z.eval();
    W_r.eval();
    W_h.eval();
    V.eval();
}

int get_input_size(std::string filename) {
    std::ifstream inputFile(filename);
    int n, inputSize = 0;
    while(!inputFile.eof()){
        inputFile >> n;
        inputSize++;
    }
    inputFile.close();
    return inputSize;
}

void read_x_y(MatrixXf& x, MatrixXf& y, std::string filename, int time_steps, int pos) {
    filename.replace(filename.end() - 3, filename.end(), "bin");
    std::ifstream file(filename, std::ios::binary);

    int count = 0;
    int n;

    file.seekg(pos * sizeof(int), std::ios::beg);
    uint32_t a = 0;

    while(!file.eof() && count < time_steps) {
        file.read((char*)&a, sizeof(uint32_t));
        x(count, int(a)) = 1;
        count++;
    }

    file.read(reinterpret_cast<char *>(&a), sizeof(a));
    y(0, int(a)) = 1;

    x.eval();
    y.eval();

    file.close();
}

float validate(MatrixXf& U_z, MatrixXf& U_r, MatrixXf& U_h, MatrixXf& W_z, MatrixXf& W_r, MatrixXf& W_h, MatrixXf& V, int input_dim, int output_dim, int hidden_dim, int time_steps) {

    std::string filename = "Inputs/trump-test-encoded.txt";
    int inputSize   = get_input_size(filename) - time_steps - 1;
    int limit       = inputSize;
    float loss      = 0;
    for(int i = 0; i < limit; i++) {
        MatrixXf E      = MatrixXf::Zero(1, time_steps);
        MatrixXf z      = MatrixXf::Zero(time_steps, hidden_dim);
        MatrixXf r      = MatrixXf::Zero(time_steps, hidden_dim);
        MatrixXf h      = MatrixXf::Zero(time_steps, hidden_dim);
        MatrixXf O      = MatrixXf::Zero(time_steps, output_dim);
        MatrixXf S      = MatrixXf::Zero(time_steps + 1, hidden_dim);
        S(0, 0)         = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX / 2.0)) - 1);
        MatrixXf currX  = MatrixXf::Zero(time_steps, input_dim);
        MatrixXf currY  = MatrixXf::Zero(1, output_dim);

        read_x_y(currX, currY, filename, time_steps, i);

        E.eval();
        z.eval();
        r.eval();
        h.eval();
        O.eval();
        S.eval();
        currX.eval();
        currY.eval();


        forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, currX, currY, O, S, E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);

        loss += (calculate_cost(E, time_steps) / limit);
    }

    return loss;
}

int train(std::string filename, float learning_rate, int nepoch, int input_dim, int hidden_dim, int output_dim, int time_steps, float decay) {

    float prev_loss = 0.0;
    float loss      = 0.0;
    int inputSize   = get_input_size(filename) - time_steps - 1;
    int limit       = inputSize;
    float min_loss  = 99999999.0;
    int batch_size  = 1;
    float test_loss = 0.0;
    std::cout << "Size of input file is : " << inputSize << std::endl;

    MatrixXf U_z, U_r, U_h, W_z, W_r, W_h, V, U_z_grad_batch, U_r_grad_batch, U_h_grad_batch, W_r_grad_batch, W_h_grad_batch, W_z_grad_batch, V_grad_batch;
    init_weight_matrices(U_z, U_r, U_h, W_z, W_r, W_h, V, input_dim, output_dim, hidden_dim);

    U_z_grad_batch = MatrixXf::Zero(input_dim, hidden_dim);
    U_r_grad_batch = MatrixXf::Zero(input_dim, hidden_dim);
    U_h_grad_batch = MatrixXf::Zero(input_dim, hidden_dim);
    W_z_grad_batch = MatrixXf::Zero(hidden_dim, hidden_dim);
    W_r_grad_batch = MatrixXf::Zero(hidden_dim, hidden_dim);
    W_h_grad_batch = MatrixXf::Zero(hidden_dim, hidden_dim);
    V_grad_batch   = MatrixXf::Zero(hidden_dim, output_dim);

    MatrixXf U_z_grad = MatrixXf::Zero(input_dim, hidden_dim);
    MatrixXf U_r_grad = MatrixXf::Zero(input_dim, hidden_dim);
    MatrixXf U_h_grad = MatrixXf::Zero(input_dim, hidden_dim);
    MatrixXf W_z_grad = MatrixXf::Zero(hidden_dim, hidden_dim);
    MatrixXf W_r_grad = MatrixXf::Zero(hidden_dim, hidden_dim);
    MatrixXf W_h_grad = MatrixXf::Zero(hidden_dim, hidden_dim);
    MatrixXf V_grad   = MatrixXf::Zero(hidden_dim, output_dim);

    U_z_grad.eval();
    U_r_grad.eval();
    U_h_grad.eval();
    W_z_grad.eval();
    W_r_grad.eval();
    W_h_grad.eval();
    V_grad.eval();

    MatrixXf cache_U_z = MatrixXf::Ones(input_dim, hidden_dim);
    MatrixXf cache_U_r = MatrixXf::Ones(input_dim, hidden_dim);
    MatrixXf cache_U_h = MatrixXf::Ones(input_dim, hidden_dim);
    MatrixXf cache_W_z = MatrixXf::Ones(hidden_dim, hidden_dim);
    MatrixXf cache_W_r = MatrixXf::Ones(hidden_dim, hidden_dim);
    MatrixXf cache_W_h = MatrixXf::Ones(hidden_dim, hidden_dim);
    MatrixXf cache_V   = MatrixXf::Ones(hidden_dim, output_dim);

    cache_U_z.eval();
    cache_U_r.eval();
    cache_U_h.eval();
    cache_W_r.eval();
    cache_W_z.eval();
    cache_V.eval();

    for(int epoch = 0; epoch < nepoch; epoch++) {

        float loss = 0;
        std::cout << "Epoch: " << epoch << std::endl;
        for(int i = 0; i < limit; i++) {
            // std::cout << i << std::endl;

            MatrixXf E      = MatrixXf::Zero(1, time_steps);
            MatrixXf z      = MatrixXf::Zero(time_steps, hidden_dim);
            MatrixXf r      = MatrixXf::Zero(time_steps, hidden_dim);
            MatrixXf h      = MatrixXf::Zero(time_steps, hidden_dim);
            MatrixXf O      = MatrixXf::Zero(time_steps, output_dim);
            MatrixXf S      = MatrixXf::Zero(time_steps + 1, hidden_dim);
            S(0, 0)         = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX / 2.0)) - 1);
            MatrixXf currX  = MatrixXf::Zero(time_steps, input_dim);
            MatrixXf currY  = MatrixXf::Zero(1, output_dim);

            read_x_y(currX, currY, filename, time_steps, i);

            E.eval();
            z.eval();
            r.eval();
            h.eval();
            O.eval();
            S.eval();
            currX.eval();
            currY.eval();

            forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, currX, currY, O, S, E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
            loss += (calculate_cost(E, time_steps) / limit);

            back_propagation(V, U_z, U_r, U_h, W_z, W_r, W_h, U_z_grad, U_r_grad, U_h_grad, W_z_grad, W_r_grad, W_h_grad, V_grad, z, r, h, O, S, E, currX, currY, input_dim, hidden_dim, output_dim, time_steps);

            // gradient_descent(U_z, U_r, U_h, W_z, W_r, W_h, V, U_z_grad, U_r_grad, U_h_grad, W_z_grad, W_r_grad, W_h_grad, V_grad, learning_rate, input_dim, hidden_dim, output_dim);
            rms_prop(U_z, U_r, U_h, W_z, W_r, W_h, V, U_z_grad, U_r_grad, U_h_grad, W_z_grad, W_r_grad, W_h_grad, V_grad, cache_U_z, cache_U_r, cache_U_h, cache_W_z, cache_W_r, cache_W_h, cache_V, learning_rate, input_dim, hidden_dim, output_dim);
        }

        U_z_grad_batch = MatrixXf::Zero(input_dim, hidden_dim);
        U_r_grad_batch = MatrixXf::Zero(input_dim, hidden_dim);
        U_h_grad_batch = MatrixXf::Zero(input_dim, hidden_dim);
        W_z_grad_batch = MatrixXf::Zero(hidden_dim, hidden_dim);
        W_r_grad_batch = MatrixXf::Zero(hidden_dim, hidden_dim);
        W_h_grad_batch = MatrixXf::Zero(hidden_dim, hidden_dim);
        V_grad_batch   = MatrixXf::Zero(hidden_dim, output_dim);

        std::cout << "Training Loss: " << loss << ", Epoch: "<< epoch << std::endl;

        test_loss = validate(U_z, U_r, U_h, W_z, W_r, W_h, V, input_dim, output_dim, hidden_dim, time_steps);

        std::cout << "Validation Loss: " << test_loss << ", Epoch: "<< epoch << std::endl;

        if(loss > prev_loss && prev_loss != 0) {
            learning_rate = learning_rate * 1;
            std::cout << "Adjusting learning rate to " << learning_rate << std::endl;
        }

        prev_loss = loss;
        learning_rate *= 1 / (1 + (decay * epoch));

        if(loss < min_loss){
            min_loss = loss;
            std::cout << "Writing weights to file. " << std::endl;
            write_binary_matrix("Weights/Uz_epoch_" + std::to_string(epoch) + "_loss_" + std::to_string(loss) + ".bin", U_z);
            write_binary_matrix("Weights/Uh_epoch_" + std::to_string(epoch) + "_loss_" + std::to_string(loss) + ".bin", U_h);
            write_binary_matrix("Weights/Ur_epoch_" + std::to_string(epoch) + "_loss_" + std::to_string(loss) + ".bin", U_r);
            write_binary_matrix("Weights/Wz_epoch_" + std::to_string(epoch) + "_loss_" + std::to_string(loss) + ".bin", W_z);
            write_binary_matrix("Weights/Wh_epoch_" + std::to_string(epoch) + "_loss_" + std::to_string(loss) + ".bin", W_h);
            write_binary_matrix("Weights/Wr_epoch_" + std::to_string(epoch) + "_loss_" + std::to_string(loss) + ".bin", W_r);
            write_binary_matrix("Weights/V_epoch_"  + std::to_string(epoch) + "_loss_" + std::to_string(loss) + ".bin", V);
        }

    }

    return 0;
}

int main(int argc, char *argv[])
{
    int input_dim       = 54;
    int hidden_dim      = 140;
    int output_dim      = 54;
    float learning_rate = 0.005;
    int nepochs         = 20000;
    int time_steps      = 20;
    float decay         = 0.001;

    int status = train("Inputs/trump-train-encoded.txt", learning_rate, nepochs, input_dim, hidden_dim, output_dim, time_steps, decay);
    return 0;
}