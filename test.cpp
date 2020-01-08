#include <iostream>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cmath>
#include <math.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <assert.h>
#include <map>

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
    if(x == 0)
        return 0.0;
    else
        return log(x);
}

float max = 0;
float softmax(float x) {
    return exp(x - max);
}

void forward_propagation_test(MatrixXf& U_z, MatrixXf& U_r, MatrixXf& U_h, MatrixXf& W_z, MatrixXf& W_r, MatrixXf& W_h, MatrixXf& V, MatrixXf& X, MatrixXf& O, MatrixXf& S, MatrixXf& z, MatrixXf& r, MatrixXf& h, int input_dim, int hidden_dim, int output_dim, int time_steps) {
    /* Forward propagation Step - returns z, r, h, S, O, E */

    MatrixXf temp           = MatrixXf::Zero(1, hidden_dim);
    MatrixXf temp_output    = MatrixXf::Zero(1, output_dim);
    MatrixXf temp_hidden    = MatrixXf::Zero(1, hidden_dim);
    for(int i = 0; i < time_steps; i++) {
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

        temp_hidden     = (MatrixXf::Ones(1, hidden_dim) - z.row(i)).cwiseProduct(h.row(i)) + z.row(i).cwiseProduct(S.row(i));
        temp_hidden.eval();

        S.row(i + 1)    = temp_hidden;//.unaryExpr(&tanh_activation);
        S.eval();
        temp_output     = S.row(i + 1) * (V);
        temp_output.eval();

        // temp_output.unaryExpr(&div_temp);
        max             = temp_output.maxCoeff();
        temp_output     = temp_output.unaryExpr(&softmax);
        temp_output.eval();

        O.row(i)        = temp_output / temp_output.sum();
        O.eval();

        temp_output     = O.row(i);
        temp_output.eval();
    }
}


int get_max_index(MatrixXf& O, std::vector<int> predictions, int time_steps) {
    int max_index;
    int index = time_steps - 1;
    O.row(index).maxCoeff(&max_index);

    return max_index;
}

void predict(MatrixXf& U_z, MatrixXf& U_r, MatrixXf& U_h, MatrixXf& W_z, MatrixXf& W_r, MatrixXf& W_h, MatrixXf& V,
          int input_dim, int output_dim, int hidden_dim, int time_steps) {

    MatrixXf X, Y, U_z_grad, U_r_grad, U_h_grad, W_z_grad, W_r_grad, W_h_grad, V_grad;

    MatrixXf z      = MatrixXf::Zero(time_steps, hidden_dim);
    MatrixXf r      = MatrixXf::Zero(time_steps, hidden_dim);
    MatrixXf h      = MatrixXf::Zero(time_steps, hidden_dim);
    MatrixXf O      = MatrixXf::Zero(time_steps, output_dim);
    MatrixXf S      = MatrixXf::Zero(time_steps + 1, hidden_dim);
    S(0, 0)         = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX / 2.0)) - 1);

    X               = MatrixXf::Zero(time_steps, input_dim);
    Y               = MatrixXf::Zero(1, output_dim);

    z.eval();
    r.eval();
    h.eval();
    O.eval();
    S.eval();

    int max_index;
    int start_count;
    char mapping_57[] = {' ', '!', '"', '&', '^', '(', ')', '+', ',', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '*', '-', '_', '`', '~', '@', '#'};
    char mapping_27[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '};
    char mapping_39[] = { '\n', ' ', '!', '$', '&', '"', ',', '-', '.', '3', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };
    char mapping_64[] = {'\n', ' ', '!', '"', '#', '$', '%', '&', '`', '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '@', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~'};

    std::map<char, int> char_to_int;

    for (int i = 0; i < 63; ++i)
    {
        char_to_int.insert(std::pair<char, int>(mapping_64[i], i));
    }

    std::string input = "i am the best";
    std::cout << "Input length = " << input.length() << std::endl;
    assert(input.length() == time_steps);

    int i = 0;
    for(auto c : input) {
        X(i, char_to_int.find(c)->second) = 1;
        i++;
    }
    std::cout << std::endl;

    std::vector<int> predictions;

    int count = 250;
    int s;

    X.eval();
    while(count --)
    {
        forward_propagation_test(U_z, U_r, U_h, W_z, W_r, W_h, V, X, O, S, z, r, h, input_dim, hidden_dim, output_dim, time_steps);
        max_index = get_max_index(O, predictions, time_steps);
        predictions.push_back(max_index);
        X  = MatrixXf::Zero(time_steps, input_dim);
        X.eval();
        s  = predictions.size();
        start_count = time_steps - 1;
        for(int i = time_steps - 1; i >= 0; i--) {
            if(s) {
                X(i, predictions[s - 1]) = 1;
                s--;
            }
            else{
                X(i, char_to_int.find(input[start_count])->second) = 1;
                start_count--;
            }
        }
    }
    std::cout << "Predictions starting with : \n" << input << std::endl;

    for(auto p : predictions){
        std::cout << mapping_64[p];
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    int time_steps      = 13;
    float decay         = 0.0;

    MatrixXf U_z, U_r, U_h, W_z, W_r, W_h, V;

    std::string epoch = argv[1];
    std::string loss  = argv[2];

    read_binary_matrix("Weights/Uz_epoch_" + epoch + "_loss_" + loss + ".bin", U_z);
    read_binary_matrix("Weights/Uh_epoch_" + epoch + "_loss_" + loss + ".bin", U_h);
    read_binary_matrix("Weights/Ur_epoch_" + epoch + "_loss_" + loss + ".bin", U_r);
    read_binary_matrix("Weights/Wz_epoch_" + epoch + "_loss_" + loss + ".bin", W_z);
    read_binary_matrix("Weights/Wh_epoch_" + epoch + "_loss_" + loss + ".bin", W_h);
    read_binary_matrix("Weights/Wr_epoch_" + epoch + "_loss_" + loss + ".bin", W_r);
    read_binary_matrix("Weights/V_epoch_"  + epoch + "_loss_" + loss + ".bin", V);

    int input_dim  = U_z.rows();
    int hidden_dim = U_z.cols();
    int output_dim = V.cols();

    std::cout << input_dim << " " << hidden_dim << " " << output_dim << std::endl;

    predict(U_z, U_r, U_h, W_z, W_r, W_h, V, input_dim, output_dim, hidden_dim, time_steps);

    return 0;
}