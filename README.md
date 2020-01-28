This is an implementation of a GRU in C++ using no libraries other than Eigen for Matrices. This is a barebones implementation that 
uncovers the complexity of backpropagation. All equations have been derived and implemented.

Create a folder for weights called `Weights`

Usage -

```
make
./train
./test `epoch_number` `loss`

e.g. ./test 66 0.397798
```