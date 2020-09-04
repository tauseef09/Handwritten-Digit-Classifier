#ifndef TRAINER_H
#define TRAINER_H

#include <NeuralNet.h>
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

using namespace std;

class Trainer : public NeuralNet
{
public:
    Trainer();
    virtual ~Trainer();
    void memoryAllocation();
    void backPropagation();
    int gradientDescent();
    void input();
    void writeMatrix(string fileName);
    void train();

protected:

private:

    //placeholders for the file names
    string trainingImageFn;
    string trainingLabelFn;
    string modelFn;
    string reportFn;

    //required file variables
    ifstream image;
    ifstream label;
    ofstream report;


    int nTraining; //number of training examples
    int epochs;    //max number of times fp and bp will run on a single example
    double learningRate;
    double momentum;
    double epsilon;

    double  *deltaW1[784 + 1];

    double *deltaW2[128 + 1], *smallDel2;

    double *smallDel3;

};

#endif // TRAINER_H
