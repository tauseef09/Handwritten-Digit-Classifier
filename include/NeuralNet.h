#ifndef NEURALNET_H
#define NEURALNET_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

class NeuralNet
{
public:
    NeuralNet();
    virtual ~NeuralNet();
    double sigmoid(double x);
    void forwardPropagation();
    double squaredError();


    virtual void abstract()=0; // making the neural network class as abstract

protected:
    //height and width of the image which is 28x28
     int width;
     int height;

     int n1;
     int n2;
     int n3;

     //Weight matrix for the input layer to the hidden layer
     //vectorImg hold the vector version of our image d
    double *w1[784 + 1], *vectorImg;

    //Weight matrix for the input layer to the hidden layer
    //a2 is g(z2)
    double *w2[128 + 1], *z2, *a2;

    //a3 is g(z3)
    double *z3, *a3;
    double expectedOutput[10 + 1]; //contains the labels

    //hold the image matrix
    int img[28 + 1][28 + 1];

private:
};

#endif // NEURALNET_H
