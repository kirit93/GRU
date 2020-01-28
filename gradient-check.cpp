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

void read_input(MatrixXd& X, std::ifstream& inputFile, int time_steps) {
    /*
    * Second while loop depends on encoding scheme - for 27 character encoding
    */
    int count = 0;
    int n;
    while(!inputFile.eof() && count < time_steps) {
        inputFile >> n;
        X(count, int(n)) = 1;
        count++;
    }
    while(count < time_steps) {
        X(count, 0) = 1;
        count ++;
    }
    X.eval();
}

void read_output(MatrixXd& Y, std::ifstream& outputFile, int time_steps) {
    int count = 0;
    int n;
    while(!outputFile.eof() && count < time_steps) {
        outputFile >> n;
        Y(count, int(n)) = 1;
        count++;
    }
    while(count < time_steps) {
        Y(count, 0) = 1;
        count ++;
    }
    Y.eval();
}

void write_binary_matrix(std::string filename, const MatrixXd& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename MatrixXd::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename MatrixXd::Index));
    out.write((char*) (&cols), sizeof(typename MatrixXd::Index));
    out.write((char*) matrix.data(), rows * cols * sizeof(typename MatrixXd::Scalar) );
    out.close();
}

void read_binary_matrix(std::string filename, MatrixXd& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename MatrixXd::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename MatrixXd::Index));
    in.read((char*) (&cols),sizeof(typename MatrixXd::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows * cols * sizeof(typename MatrixXd::Scalar) );
    in.close();
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_grad(double x) {
    return (1 - x) * x;
}

double relu(double x) {
    if(x <= 0)
        return 0.001 * x;
    else
        return x;
}

double relu_grad(double x) {
   if(x <= 0)
        return 0.001;
    else
        return 1;
}

/* Tanh activation function */
double tanh_activation(double x) {
    return tanh(x);
}

/* Tanh of sigmoid function */
double tanh_grad(double x) {
    return 1 - (x * x);
}

double log_matrix(double x) {
    if(x <= 0)
        return 0.0;
    else
        return log(x);
}

double max = 0;
double softmax(double x) {
    return exp(x - max);
}

MatrixXd loss_softmax_grad(MatrixXd& y, MatrixXd& o) {
    return o - y;
}

void dropout(MatrixXd& m, double p) {
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

double div_temp(double x) {
    double temp = 0.5;
    if(x <= 0)
        x = 0;
    else
        x = log(x) / temp;
    return x;
}
void forward_propagation(MatrixXd& U_z, MatrixXd& U_r, MatrixXd& U_h, MatrixXd& W_z, MatrixXd& W_r, MatrixXd& W_h, MatrixXd& V, MatrixXd& X, MatrixXd& Y, MatrixXd& O, MatrixXd& S, MatrixXd& E, MatrixXd& z, MatrixXd& r, MatrixXd& h, int time_steps, int input_dim, int hidden_dim, int output_dim) {
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

    MatrixXd temp           = MatrixXd::Zero(1, hidden_dim);
    MatrixXd temp_output    = MatrixXd::Zero(1, output_dim);
    MatrixXd temp_hidden    = MatrixXd::Zero(1, hidden_dim);
    for(int i = 0; i < time_steps; i++) {

        // dropout(S, 0.2);

        temp            = (X.row(i) * (U_z)) + (S.row(i) * (W_z));
        temp.eval();
        z.row(i)        = temp.unaryExpr(&sigmoid);
        z.eval();

        temp            = (X.row(i) * (U_r)) + (S.row(i) * (W_r));
        temp.eval();
        r.row(i)        = temp.unaryExpr(&sigmoid);
        r.eval();

        temp            = (X.row(i) * (U_h)) + (S.row(i).cwiseProduct(r.row(i))) * (W_h);
        temp.eval();
        h.row(i)        = temp.unaryExpr(&tanh_activation);
        h.eval();

        temp_hidden     = (MatrixXd::Ones(1, hidden_dim) - z.row(i)).cwiseProduct(h.row(i)) + z.row(i).cwiseProduct(S.row(i));
        temp_hidden.eval();
        S.row(i + 1)    = temp_hidden;//.unaryExpr(&sigmoid);
        S.eval();

        temp_output     = S.row(i + 1) * (V);
        temp_output.eval();

        // temp_output.unaryExpr(&div_temp);

        max             = temp_output.maxCoeff();
        temp_output     = temp_output.unaryExpr(&softmax);
        O.row(i)        = temp_output / temp_output.sum();
        O.eval();

        temp_output     = O.row(i);
        temp_output.eval();

        // E(0, i)         += -1 * (Y.row(0).cwiseProduct(temp_output.unaryExpr(&log_matrix)).sum());
        // E.eval();
    }

    E(0, 0)         = -1 * (Y.row(0).cwiseProduct(temp_output.unaryExpr(&log_matrix)).sum());
    E.eval();
}

double calculate_cost(MatrixXd& E) {
    /* Possibly - Move to forward propagation or Train */
    return E.sum();//(E.sum() / time_steps);
}

void two_layer_back_propagation(MatrixXd& delta_next, MatrixXd& V, MatrixXd& W_z, MatrixXd& W_r, MatrixXd& W_h, MatrixXd& dU_z, MatrixXd& dU_r, MatrixXd& dU_h, MatrixXd& dW_z, MatrixXd& dW_r, MatrixXd& dW_h, MatrixXd& dV, MatrixXd& z, MatrixXd& r, MatrixXd& h, MatrixXd& O, MatrixXd& S, MatrixXd& X, MatrixXd& Y, int input_dim, int hidden_dim, int output_dim, int time_steps) {
    /* gradients = dLdV, dLdU0, dLdU1, dLdU2, dLdW0, dLdW1, dLdW2 */
    int time_step;
    dU_z = MatrixXd::Zero(input_dim, hidden_dim);
    dU_r = MatrixXd::Zero(input_dim, hidden_dim);
    dU_h = MatrixXd::Zero(input_dim, hidden_dim);
    dW_z = MatrixXd::Zero(hidden_dim, hidden_dim);
    dW_r = MatrixXd::Zero(hidden_dim, hidden_dim);
    dW_h = MatrixXd::Zero(hidden_dim, hidden_dim);

    dU_z.eval();
    dU_r.eval();
    dU_h.eval();
    dW_z.eval();
    dW_r.eval();
    dW_h.eval();

    MatrixXd ds_0, temp_o, dsr, ds_single, ds_cur, delta_y, ds_cur_bk, dz, db_V, dreluInput_z, dreluInput_r, dreluInput_h, temp_r, temp_z, temp_h, temp_S, temp_X, temp_W, temp_U, temp_V;
    dsr = MatrixXd::Zero(1, hidden_dim);
    temp_S = MatrixXd::Zero(1, hidden_dim);
    temp_z = MatrixXd::Zero(1, hidden_dim);
    temp_r = MatrixXd::Zero(1, hidden_dim);
    temp_h = MatrixXd::Zero(1, hidden_dim);
    temp_X = MatrixXd::Zero(1, input_dim);
    ds_cur_bk = MatrixXd::Zero(1, hidden_dim);
    delta_y = delta_next;

    dsr.eval();
    temp_S.eval();
    temp_z.eval();
    temp_r.eval();
    temp_h.eval();
    temp_X.eval();
    ds_cur_bk.eval();
    delta_y.eval();

    dV = MatrixXd::Zero(hidden_dim, output_dim);
    dV.eval();

    temp_S = S.row(time_steps);
    temp_S.eval();

    dV = temp_S.transpose().eval() * delta_y;
    dV.eval();

    ds_single = delta_y * V.transpose().eval();
    ds_single.eval();

    ds_cur = ds_single;
    ds_cur.eval();

    for(time_step = time_steps - 1; time_step >= 0; time_step--) {
        ds_cur_bk = ds_cur;
        temp_S = S.row(time_step);
        temp_r = r.row(time_step);
        temp_z = z.row(time_step);
        temp_h = h.row(time_step);
        temp_X = X.row(time_step);

        dreluInput_h = ds_cur.cwiseProduct(MatrixXd::Ones(1, hidden_dim) - temp_z).cwiseProduct(temp_h.unaryExpr(&tanh_grad));//.cwiseProduct(temp_S)).unaryExpr(&tanh_grad));
        dreluInput_h.eval();

        temp_U = (temp_X.transpose().eval() * dreluInput_h).eval();
        dU_h = dU_h + temp_U;
        dU_h.eval();

        temp_W = ((temp_S.cwiseProduct(temp_r)).transpose().eval() * dreluInput_h).eval();
        dW_h = dW_h + temp_W;
        dW_h.eval();

        dsr = dreluInput_h * W_h.transpose().eval();
        dsr.eval();

        ds_cur = dsr.cwiseProduct(temp_r);
        dreluInput_r = dsr.cwiseProduct(temp_S).cwiseProduct(temp_r.unaryExpr(&sigmoid_grad));
        dreluInput_r.eval();

        temp_U = (temp_X.transpose().eval() * dreluInput_r);
        temp_U.eval();

        dU_r = dU_r + temp_U;
        dU_r.eval();

        temp_W = (temp_S.transpose().eval() * dreluInput_r);
        temp_W.eval();

        dW_r = dW_r + temp_W;
        dW_r.eval();

        ds_cur = ds_cur + (dreluInput_r * W_r.transpose().eval());
        ds_cur.eval();

        ds_cur = ds_cur + ds_cur_bk.cwiseProduct(temp_z);
        ds_cur.eval();

        dz = ds_cur_bk.cwiseProduct(temp_S - temp_h);
        dz.eval();

        dreluInput_z = dz.cwiseProduct(temp_z.unaryExpr(&sigmoid));
        dreluInput_z.eval();

        temp_U = (temp_X.transpose().eval() * dreluInput_z);
        temp_U.eval();

        dU_z = dU_z + temp_U;
        dU_z.eval();

        temp_W = (temp_S.transpose().eval() * dreluInput_z);
        temp_W.eval();

        dW_z = dW_z + temp_W;
        dW_z.eval();

        ds_cur = ds_cur + (dreluInput_z * W_z.transpose().eval());
        ds_cur.eval();

        // temp_o = dreluInput_z * U_z.transpose().eval();
        // delta_next.row(time_step) = temp_o;

    }

    delta_next = ds_cur;

    dU_z /= time_steps;
    dU_r /= time_steps;
    dU_h /= time_steps;
    dW_z /= time_steps;
    dW_r /= time_steps;
    dW_h /= time_steps;
    dV   /= time_steps;

    // std::cout << "Vgrad other = " << dV << std::endl;

    dU_z.eval();
    dU_r.eval();
    dU_h.eval();
    dW_z.eval();
    dW_r.eval();
    dW_h.eval();
    dV.eval();
}

void init_matrix(MatrixXd& X, double dimension_row, double dimension_col) {
    double upperlimit = 1.0 * sqrt(1.0 / (double)dimension_row);
    double lowerlimit = -1.0 * sqrt(1.0 / (double)dimension_row);;
    double range = upperlimit - lowerlimit;

    srand(clock());
    X = MatrixXd::Random(dimension_row, dimension_col);
    X = (X + MatrixXd::Constant(dimension_row, dimension_col, 1.0)) * (range / 2.0);
    X = (X + MatrixXd::Constant(dimension_row, dimension_col, lowerlimit));
}

void init_weight_matrices(MatrixXd& U_z, MatrixXd& U_r, MatrixXd& U_h, MatrixXd& W_z, MatrixXd& W_r, MatrixXd& W_h, MatrixXd& V, int input_dim, int output_dim, int hidden_dim) {
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
    ;
    inputFile.close();
    return inputSize;
}

void read_x_y(MatrixXd& x, MatrixXd& y, std::string filename, int time_steps, int pos) {
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

int train(std::string filename, double learning_rate, int nepoch, int input_dim, int hidden_dim, int output_dim, int time_steps, double decay) {

    double prev_loss = 0.0;
    double loss      = 0.0;
    int inputSize   = get_input_size(filename) - time_steps - 1;
    int limit       = 100;//inputSize / 100;
    double min_loss  = 99999999.0;
    int batch_size  = 1;

    std::cout << "Size of input file is : " << inputSize << std::endl;

    MatrixXd U_z, U_r, U_h, W_z, W_r, W_h, V, U_z_grad_batch, U_r_grad_batch, U_h_grad_batch, W_r_grad_batch, W_h_grad_batch, W_z_grad_batch, V_grad_batch;
    init_weight_matrices(U_z, U_r, U_h, W_z, W_r, W_h, V, input_dim, output_dim, hidden_dim);

    MatrixXd U_z2, U_r2, U_h2, W_z2, W_r2, W_h2, V2;
    MatrixXd delta_next;
    init_weight_matrices(U_z2, U_r2, U_h2, W_z2, W_r2, W_h2, V2, input_dim, output_dim, hidden_dim);

    MatrixXd U_z_grad = MatrixXd::Zero(input_dim, hidden_dim);
    MatrixXd U_r_grad = MatrixXd::Zero(input_dim, hidden_dim);
    MatrixXd U_h_grad = MatrixXd::Zero(input_dim, hidden_dim);
    MatrixXd W_z_grad = MatrixXd::Zero(hidden_dim, hidden_dim);
    MatrixXd W_r_grad = MatrixXd::Zero(hidden_dim, hidden_dim);
    MatrixXd W_h_grad = MatrixXd::Zero(hidden_dim, hidden_dim);
    MatrixXd V_grad   = MatrixXd::Zero(hidden_dim, output_dim);

    MatrixXd U_z_grad2 = MatrixXd::Zero(input_dim, hidden_dim);
    MatrixXd U_r_grad2 = MatrixXd::Zero(input_dim, hidden_dim);
    MatrixXd U_h_grad2 = MatrixXd::Zero(input_dim, hidden_dim);
    MatrixXd W_z_grad2 = MatrixXd::Zero(hidden_dim, hidden_dim);
    MatrixXd W_r_grad2 = MatrixXd::Zero(hidden_dim, hidden_dim);
    MatrixXd W_h_grad2 = MatrixXd::Zero(hidden_dim, hidden_dim);
    MatrixXd V_grad2   = MatrixXd::Zero(hidden_dim, output_dim);

    U_z_grad.eval();
    U_r_grad.eval();
    U_h_grad.eval();
    W_z_grad.eval();
    W_r_grad.eval();
    W_h_grad.eval();
    V_grad.eval();

    U_z_grad2.eval();
    U_r_grad2.eval();
    U_h_grad2.eval();
    W_z_grad2.eval();
    W_r_grad2.eval();
    W_h_grad2.eval();
    V_grad2.eval();

    int i = 0;

    MatrixXd E      = MatrixXd::Zero(1, time_steps);
    // MatrixXd E      = MatrixXd::Zero(1, 1);
    MatrixXd z      = MatrixXd::Zero(time_steps, hidden_dim);
    MatrixXd r      = MatrixXd::Zero(time_steps, hidden_dim);
    MatrixXd h      = MatrixXd::Zero(time_steps, hidden_dim);
    MatrixXd O      = MatrixXd::Zero(time_steps, output_dim);
    MatrixXd S      = MatrixXd::Zero(time_steps + 1, hidden_dim);

    MatrixXd z2      = MatrixXd::Zero(time_steps, hidden_dim);
    MatrixXd r2      = MatrixXd::Zero(time_steps, hidden_dim);
    MatrixXd h2      = MatrixXd::Zero(time_steps, hidden_dim);
    MatrixXd O2      = MatrixXd::Zero(time_steps, output_dim);
    MatrixXd S2      = MatrixXd::Zero(time_steps + 1, hidden_dim);

    S(0, 0)          = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX / 2.0)) - 1);
    S2(0, 0)         = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX / 2.0)) - 1);

    MatrixXd currX  = MatrixXd::Zero(time_steps, input_dim);
    MatrixXd currY  = MatrixXd::Zero(1, output_dim);

    read_x_y(currX, currY, filename, time_steps, i);

    E.eval();
    z.eval();
    r.eval();
    h.eval();
    O.eval();
    S.eval();

    z2.eval();
    r2.eval();
    h2.eval();
    O2.eval();
    S2.eval();
    currX.eval();
    currY.eval();

    forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, currX, currY, O, S, E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);

    E      = MatrixXd::Zero(1, time_steps);
    E.eval();
    forward_propagation(U_z2, U_r2, U_h2, W_z2, W_r2, W_h2, V2, O, currY, O2, S2, E, z2, r2, h2, time_steps, input_dim, hidden_dim, output_dim);

    loss += (calculate_cost(E) / limit);

    // back_propagation(V, W_h, U_z_grad, U_r_grad, U_h_grad, W_z_grad, W_r_grad, W_h_grad, V_grad, z, r, h, O, S, currX, currY, input_dim, hidden_dim, output_dim, time_steps);
    // other_back_propagation(V, W_z, W_r, W_h, U_z_grad, U_r_grad, U_h_grad, W_z_grad, W_r_grad, W_h_grad, V_grad, z, r, h, O, S, currX, currY, input_dim, hidden_dim, output_dim, time_steps);

    delta_next = O2.row(time_steps - 1) - currY.row(0);
    two_layer_back_propagation(delta_next, V2, W_z2, W_r2, W_h2, U_z_grad2, U_r_grad2, U_h_grad2, W_z_grad2, W_r_grad2, W_h_grad2, V_grad2, z2, r2, h2, O2, S2, O, currY, input_dim, hidden_dim, output_dim, time_steps);

    delta_next = O.row(time_steps - 1);
    two_layer_back_propagation(delta_next, V, W_z, W_r, W_h, U_z_grad, U_r_grad, U_h_grad, W_z_grad, W_r_grad, W_h_grad, V_grad, z, r, h, O, S, currX, currY, input_dim, hidden_dim, output_dim, time_steps);

    U_z_grad = U_z_grad.cwiseProduct(U_z_grad2);
    U_r_grad = U_r_grad.cwiseProduct(U_r_grad2);
    U_h_grad = U_h_grad.cwiseProduct(U_h_grad2);
    W_z_grad = W_z_grad.cwiseProduct(W_z_grad2);
    W_r_grad = W_r_grad.cwiseProduct(W_r_grad2);
    W_h_grad = W_h_grad.cwiseProduct(W_h_grad2);
    V_grad   = V_grad.cwiseProduct(V_grad2);

    U_z_grad.eval();
    U_r_grad.eval();
    U_h_grad.eval();
    W_z_grad.eval();
    W_r_grad.eval();
    W_h_grad.eval();
    V_grad.eval();

    int count = 0, count_errors = 0;
    double temp, backup, eps = 0.0001, calc_grad, temp_loss_1, temp_loss_2, max_error = 0.0;
    for(int k = 0; k < W_h.rows(); k++) {
        for(int j = 0; j < W_h.cols(); j++) {
            temp = W_h(k, j);
            backup = temp;
            temp += eps;
            W_h(k, j) = temp;
            W_h.eval();
            forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, currX, currY, O, S, E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
            E.eval();
            temp_loss_1 = (calculate_cost(E) / time_steps);

            temp = backup;
            temp -= eps;
            W_h(k, j) = temp;
            W_h.eval();
            forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, currX, currY, O, S, E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
            E.eval();
            temp_loss_2 = (calculate_cost(E) / time_steps);

            W_h(k, j) = backup;
            calc_grad = (temp_loss_1 - temp_loss_2) / (2 * eps);
            W_h_grad(k, j) = calc_grad;
        }
    }

    std::cout << count_errors << " out of " << count << std::endl;
    std::cout << "Max_error = " << max_error << std::endl;

    U_z_grad_batch += U_z_grad;
    U_r_grad_batch += U_r_grad;
    U_h_grad_batch += U_h_grad;
    W_z_grad_batch += W_z_grad;
    W_r_grad_batch += W_r_grad;
    W_h_grad_batch += W_h_grad;
    V_grad_batch   += V_grad;

    return 0;
}

int main(int argc, char *argv[])
{
    int input_dim        = 57;
    int hidden_dim       = 57;
    int output_dim       = 57;
    double learning_rate = 0.005;
    int nepochs          = 1000;
    int time_steps       = 5;
    double decay         = 0.0;

    int status = train("Inputs/encoded-input-test.txt", learning_rate, nepochs, input_dim, hidden_dim, output_dim, time_steps, decay);
    return 0;
}

// void back_propagation(MatrixXf& V, MatrixXf& U_z, MatrixXf& U_r, MatrixXf&
// U_h, MatrixXf& W_z, MatrixXf& W_r, MatrixXf& W_h, MatrixXf& dU_z, MatrixXf&
// dU_r, MatrixXf& dU_h, MatrixXf& dW_z, MatrixXf& dW_r, MatrixXf& dW_h,
// MatrixXf& dV, MatrixXf& z, MatrixXf& r, MatrixXf& h, MatrixXf& O, MatrixXf&
// S, MatrixXf& E, MatrixXf& X, MatrixXf& Y, int input_dim, int hidden_dim, int
// output_dim, int time_steps) {
//     /* gradients = dLdV, dLdU0, dLdU1, dLdU2, dLdW0, dLdW1, dLdW2 */
//     int time_step;
//     dU_z = MatrixXf::Zero(input_dim, hidden_dim);
//     dU_r = MatrixXf::Zero(input_dim, hidden_dim);
//     dU_h = MatrixXf::Zero(input_dim, hidden_dim);
//     dW_z = MatrixXf::Zero(hidden_dim, hidden_dim);
//     dW_r = MatrixXf::Zero(hidden_dim, hidden_dim);
//     dW_h = MatrixXf::Zero(hidden_dim, hidden_dim);
//     dV   = MatrixXf::Zero(hidden_dim, output_dim);

//     float temp, backup, eps = 0.0001, calc_grad, temp_loss_1, temp_loss_2;
//     temp = 0, backup = 0, calc_grad = 0, temp_loss_1 = 0, temp_loss_1 = 0;
//     for(int k = 0; k < V.rows(); k++) {
//         for(int j = 0; j < V.cols(); j++) {
//             temp = V(k, j);
//             backup = temp;
//             temp += eps;
//             V(k, j) = temp;
//             V.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_1 = (calculate_cost(E, time_steps));

//             temp = backup;
//             temp -= eps;
//             V(k, j) = temp;
//             V.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_2 = (calculate_cost(E, time_steps));

//             V(k, j) = backup;
//             calc_grad = (temp_loss_1 - temp_loss_2) / (2 * eps);
//             dV(k, j) = calc_grad;
//         }
//     }

//     temp = 0, backup = 0, calc_grad = 0, temp_loss_1 = 0, temp_loss_1 = 0;
//     for(int k = 0; k < U_z.rows(); k++) {
//         for(int j = 0; j < U_z.cols(); j++) {
//             temp = U_z(k, j);
//             backup = temp;
//             temp += eps;
//             U_z(k, j) = temp;
//             U_z.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_1 = (calculate_cost(E, time_steps));

//             temp = backup;
//             temp -= eps;
//             U_z(k, j) = temp;
//             U_z.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_2 = (calculate_cost(E, time_steps));

//             U_z(k, j) = backup;
//             calc_grad = (temp_loss_1 - temp_loss_2) / (2 * eps);
//             dU_z(k, j) = calc_grad;
//         }
//     }

//     temp = 0, backup = 0, calc_grad = 0, temp_loss_1 = 0, temp_loss_1 = 0;
//     for(int k = 0; k < U_r.rows(); k++) {
//         for(int j = 0; j < U_r.cols(); j++) {
//             temp = U_r(k, j);
//             backup = temp;
//             temp += eps;
//             U_r(k, j) = temp;
//             U_r.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_1 = (calculate_cost(E, time_steps));

//             temp = backup;
//             temp -= eps;
//             U_r(k, j) = temp;
//             U_r.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_2 = (calculate_cost(E, time_steps));

//             U_r(k, j) = backup;
//             calc_grad = (temp_loss_1 - temp_loss_2) / (2 * eps);
//             dU_r(k, j) = calc_grad;
//         }
//     }

//     temp = 0, backup = 0, calc_grad = 0, temp_loss_1 = 0, temp_loss_1 = 0;
//     for(int k = 0; k < U_h.rows(); k++) {
//         for(int j = 0; j < U_h.cols(); j++) {
//             temp = U_h(k, j);
//             backup = temp;
//             temp += eps;
//             U_h(k, j) = temp;
//             U_h.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_1 = (calculate_cost(E, time_steps));

//             temp = backup;
//             temp -= eps;
//             U_h(k, j) = temp;
//             U_h.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_2 = (calculate_cost(E, time_steps));

//             U_h(k, j) = backup;
//             calc_grad = (temp_loss_1 - temp_loss_2) / (2 * eps);
//             dU_h(k, j) = calc_grad;
//         }
//     }

//     temp = 0, backup = 0, calc_grad = 0, temp_loss_1 = 0, temp_loss_1 = 0;
//     for(int k = 0; k < W_z.rows(); k++) {
//         for(int j = 0; j < W_z.cols(); j++) {
//             temp = W_z(k, j);
//             backup = temp;
//             temp += eps;
//             W_z(k, j) = temp;
//             W_z.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_1 = (calculate_cost(E, time_steps));

//             temp = backup;
//             temp -= eps;
//             W_z(k, j) = temp;
//             W_z.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_2 = (calculate_cost(E, time_steps));

//             W_z(k, j) = backup;
//             calc_grad = (temp_loss_1 - temp_loss_2) / (2 * eps);
//             dW_z(k, j) = calc_grad;
//         }
//     }

//     temp = 0, backup = 0, calc_grad = 0, temp_loss_1 = 0, temp_loss_1 = 0;
//     for(int k = 0; k < W_r.rows(); k++) {
//         for(int j = 0; j < W_r.cols(); j++) {
//             temp = W_r(k, j);
//             backup = temp;
//             temp += eps;
//             W_r(k, j) = temp;
//             W_r.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_1 = (calculate_cost(E, time_steps));

//             temp = backup;
//             temp -= eps;
//             W_r(k, j) = temp;
//             W_r.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_2 = (calculate_cost(E, time_steps));

//             W_r(k, j) = backup;
//             calc_grad = (temp_loss_1 - temp_loss_2) / (2 * eps);
//             dW_r(k, j) = calc_grad;
//         }
//     }

//     temp = 0, backup = 0, calc_grad = 0, temp_loss_1 = 0, temp_loss_1 = 0;
//     for(int k = 0; k < W_h.rows(); k++) {
//         for(int j = 0; j < W_h.cols(); j++) {
//             temp = W_h(k, j);
//             backup = temp;
//             temp += eps;
//             W_h(k, j) = temp;
//             W_h.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_1 = (calculate_cost(E, time_steps));

//             temp = backup;
//             temp -= eps;
//             W_h(k, j) = temp;
//             W_h.eval();
//             forward_propagation(U_z, U_r, U_h, W_z, W_r, W_h, V, X, Y, O, S,
//             E, z, r, h, time_steps, input_dim, hidden_dim, output_dim);
//             E.eval();
//             temp_loss_2 = (calculate_cost(E, time_steps));

//             W_h(k, j) = backup;
//             calc_grad = (temp_loss_1 - temp_loss_2) / (2 * eps);
//             dW_h(k, j) = calc_grad;
//         }
//     }

//     dU_z /= time_steps;
//     dU_r /= time_steps;
//     dU_h /= time_steps;
//     dW_z /= time_steps;
//     dW_r /= time_steps;
//     dW_h /= time_steps;
//     dV /= time_steps;

//     dU_z.eval();
//     dU_r.eval();
//     dU_h.eval();
//     dW_z.eval();
//     dW_r.eval();
//     dW_h.eval();
//     dV.eval();
// }
