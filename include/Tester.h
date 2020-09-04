#ifndef TESTER_H
#define TESTER_H

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

class Tester : public NeuralNet
{
public:
    Tester();
    virtual ~Tester();
    void memoryAllocation();
    void loadModel(string fileName);
    int input();
    void test();

protected:
    string modelFn;

private:

    string testingImageFn;
    string testingLabelFn;
    string reportFn;

    ifstream image;
    ifstream label;
    ofstream report;

    int nTesting; // Number of testing samples
};

#endif // TESTER_H
