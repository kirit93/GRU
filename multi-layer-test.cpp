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

void forward_propagation_test(MatrixXf& U_z, MatrixXf& U_r, MatrixXf& U_h, MatrixXf& W_z, MatrixXf& W_r, MatrixXf& W_h, MatrixXf& V, MatrixXf& X, MatrixXf& O, MatrixXf& S, MatrixXf& z, MatrixXf& r, MatrixXf& h, int time_steps, int input_dim, int hidden_dim, int output_dim) {
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
        S.row(i + 1)    = temp_hidden;
        S.eval();

        temp_output     = S.row(i + 1) * (V);
        temp_output.eval();

        O.row(i)        = temp_output;
        O.eval();
    }
}

void softmax_layer(MatrixXf& O, int time_steps, int output_dim) {
    MatrixXf temp_output    = MatrixXf::Zero(1, output_dim);
    MatrixXf temp_ret       = MatrixXf::Zero(time_steps, output_dim);
    for(int i = 0; i < time_steps; i++) {
        temp_output = O.row(i);
        max             = temp_output.maxCoeff();
        temp_output     = temp_output.unaryExpr(&softmax);

        temp_output.eval();

        O.row(i) = temp_output / temp_output.sum();
        O.eval();
    }
}


int get_max_index(MatrixXf& O, std::vector<int> predictions, int time_steps) {
    int max_index;
    int index = time_steps - 1;
    O.row(index).maxCoeff(&max_index);

    return max_index;
}

void predict(MatrixXf& U_z_1, MatrixXf& U_r_1, MatrixXf& U_h_1, MatrixXf& W_z_1, MatrixXf& W_r_1, MatrixXf& W_h_1, MatrixXf& V_1,
            MatrixXf& U_z_2, MatrixXf& U_r_2, MatrixXf& U_h_2, MatrixXf& W_z_2, MatrixXf& W_r_2, MatrixXf& W_h_2, MatrixXf& V_2,
            MatrixXf& U_z_3, MatrixXf& U_r_3, MatrixXf& U_h_3, MatrixXf& W_z_3, MatrixXf& W_r_3, MatrixXf& W_h_3, MatrixXf& V_3,
            int input_dim, int output_dim, int hidden_dim_1, int hidden_dim_2, int hidden_dim_3, int transition_dim_1, int transition_dim_2, int time_steps) {

    MatrixXf X, Y;

    MatrixXf z_1      = MatrixXf::Zero(time_steps, hidden_dim_1);
    MatrixXf r_1      = MatrixXf::Zero(time_steps, hidden_dim_1);
    MatrixXf h_1      = MatrixXf::Zero(time_steps, hidden_dim_1);
    MatrixXf O_1      = MatrixXf::Zero(time_steps, transition_dim_1);
    MatrixXf S_1      = MatrixXf::Zero(time_steps + 1, hidden_dim_1);
    S_1(0, 0)         = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX / 2.0)) - 1);

    MatrixXf z_2      = MatrixXf::Zero(time_steps, hidden_dim_2);
    MatrixXf r_2      = MatrixXf::Zero(time_steps, hidden_dim_2);
    MatrixXf h_2      = MatrixXf::Zero(time_steps, hidden_dim_2);
    MatrixXf O_2      = MatrixXf::Zero(time_steps, transition_dim_2);
    MatrixXf S_2      = MatrixXf::Zero(time_steps + 1, hidden_dim_2);
    S_2(0, 0)         = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX / 2.0)) - 1);

    MatrixXf z_3      = MatrixXf::Zero(time_steps, hidden_dim_3);
    MatrixXf r_3      = MatrixXf::Zero(time_steps, hidden_dim_3);
    MatrixXf h_3      = MatrixXf::Zero(time_steps, hidden_dim_3);
    MatrixXf O_3      = MatrixXf::Zero(time_steps, output_dim);
    MatrixXf S_3      = MatrixXf::Zero(time_steps + 1, hidden_dim_3);
    S_3(0, 0)         = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX / 2.0)) - 1);

    X                 = MatrixXf::Zero(time_steps, input_dim);
    Y                 = MatrixXf::Zero(1, output_dim);

    z_1.eval();
    r_1.eval();
    h_1.eval();
    O_1.eval();
    S_1.eval();

    z_2.eval();
    r_2.eval();
    h_2.eval();
    O_2.eval();
    S_2.eval();

    z_3.eval();
    r_3.eval();
    h_3.eval();
    O_3.eval();
    S_3.eval();

    int max_index;
    int start_count;
    char mapping_57[] = {' ', '!', '"', '&', '^', '(', ')', '+', ',', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '*', '-', '_', '`', '~', '@', '#'};
    char mapping_27[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '};
    char mapping_39[] = { '\n', ' ', '!', '$', '&', '"', ',', '-', '.', '3', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };

    std::map<char, int> char_to_int;

    for (int i = 0; i < 39; ++i)
    {
        char_to_int.insert(std::pair<char, int>(mapping_39[i], i));
    }

    // for(auto elem : char_to_int)
    // {
    //    std::cout << elem.first << " - " << elem.second << std::endl;
    // }

    std::string input = "hello brutus, how ar";
    std::cout << "Input length = " << input.length() << std::endl;
    assert(input.length() == time_steps);


    int i = 0;
    for(auto c : input) {
        // std::cout << c << " ";
        // std::cout << char_to_int.find(c)->second << std::endl;
        X(i, char_to_int.find(c)->second) = 1;
        i++;
    }
    std::cout << std::endl;

    std::vector<int> predictions;

    int count = 1000;
    int s;

    X.eval();
    while(count --)
    {
        forward_propagation_test(U_z_1, U_r_1, U_h_1, W_z_1, W_r_1, W_h_1, V_1, X, O_1, S_1, z_1, r_1, h_1, time_steps, input_dim, hidden_dim_1, transition_dim_1);
        forward_propagation_test(U_z_2, U_r_2, U_h_2, W_z_2, W_r_2, W_h_2, V_2, O_1, O_2, S_2, z_2, r_2, h_2, time_steps, transition_dim_1, hidden_dim_2, transition_dim_2);
        forward_propagation_test(U_z_3, U_r_3, U_h_3, W_z_3, W_r_3, W_h_3, V_3, O_2, O_3, S_3, z_3, r_3, h_3, time_steps, transition_dim_2, hidden_dim_3, output_dim);
        softmax_layer(O_3, time_steps, output_dim);

        max_index = get_max_index(O_3, predictions, time_steps);
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
        std::cout << mapping_57[p];
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    float learning_rate = 0.005;
    int time_steps      = 20;
    float decay         = 0.0;

    MatrixXf U_z_1, U_r_1, U_h_1, W_z_1, W_r_1, W_h_1, V_1, U_z_2, U_r_2, U_h_2, W_z_2, W_r_2, W_h_2, V_2, U_z_3, U_r_3, U_h_3, W_z_3, W_r_3, W_h_3, V_3;

    std::string epoch = argv[1];
    std::string loss  = argv[2];

    read_binary_matrix("Weights/Uz1_epoch_" + epoch + "_loss_" + loss + ".bin", U_z_1);
    read_binary_matrix("Weights/Uh1_epoch_" + epoch + "_loss_" + loss + ".bin", U_h_1);
    read_binary_matrix("Weights/Ur1_epoch_" + epoch + "_loss_" + loss + ".bin", U_r_1);
    read_binary_matrix("Weights/Wz1_epoch_" + epoch + "_loss_" + loss + ".bin", W_z_1);
    read_binary_matrix("Weights/Wh1_epoch_" + epoch + "_loss_" + loss + ".bin", W_h_1);
    read_binary_matrix("Weights/Wr1_epoch_" + epoch + "_loss_" + loss + ".bin", W_r_1);
    read_binary_matrix("Weights/V1_epoch_"  + epoch + "_loss_" + loss + ".bin", V_1);

    read_binary_matrix("Weights/Uz2_epoch_" + epoch + "_loss_" + loss + ".bin", U_z_2);
    read_binary_matrix("Weights/Uh2_epoch_" + epoch + "_loss_" + loss + ".bin", U_h_2);
    read_binary_matrix("Weights/Ur2_epoch_" + epoch + "_loss_" + loss + ".bin", U_r_2);
    read_binary_matrix("Weights/Wz2_epoch_" + epoch + "_loss_" + loss + ".bin", W_z_2);
    read_binary_matrix("Weights/Wh2_epoch_" + epoch + "_loss_" + loss + ".bin", W_h_2);
    read_binary_matrix("Weights/Wr2_epoch_" + epoch + "_loss_" + loss + ".bin", W_r_2);
    read_binary_matrix("Weights/V2_epoch_"  + epoch + "_loss_" + loss + ".bin", V_2);

    read_binary_matrix("Weights/Uz3_epoch_" + epoch + "_loss_" + loss + ".bin", U_z_3);
    read_binary_matrix("Weights/Uh3_epoch_" + epoch + "_loss_" + loss + ".bin", U_h_3);
    read_binary_matrix("Weights/Ur3_epoch_" + epoch + "_loss_" + loss + ".bin", U_r_3);
    read_binary_matrix("Weights/Wz3_epoch_" + epoch + "_loss_" + loss + ".bin", W_z_3);
    read_binary_matrix("Weights/Wh3_epoch_" + epoch + "_loss_" + loss + ".bin", W_h_3);
    read_binary_matrix("Weights/Wr3_epoch_" + epoch + "_loss_" + loss + ".bin", W_r_3);
    read_binary_matrix("Weights/V3_epoch_"  + epoch + "_loss_" + loss + ".bin", V_3);

    int input_dim           = U_z_1.rows();
    int hidden_dim_1        = U_z_1.cols();
    int transition_dim_1    = V_1.cols();
    int hidden_dim_2        = U_z_2.cols();
    int transition_dim_2    = V_2.cols();
    int hidden_dim_3        = U_z_3.cols();
    int output_dim          = V_3.cols();

    std::cout << input_dim << " " << hidden_dim_1 << " " << transition_dim_1 << " " << hidden_dim_2 << " " << transition_dim_2 << " " << hidden_dim_3 << " " << output_dim << std::endl;

    predict(U_z_1, U_r_1, U_h_1, W_z_1, W_r_1, W_h_1, V_1, U_z_2, U_r_2, U_h_2, W_z_2, W_r_2, W_h_2, V_2, U_z_3, U_r_3, U_h_3, W_z_3, W_r_3, W_h_3, V_3, input_dim, output_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, transition_dim_1, transition_dim_2, time_steps);

    return 0;
}
