#include "NeuralNet.h"

NeuralNet::NeuralNet()
{
    //ctor
    width = 28;
    height = 28;

    n1 = width * height; //number of input neurons
    n2 = 128; //number of hidden neurons
    n3 = 10; //number of output neurons
}

NeuralNet::~NeuralNet()
{
    //dtor
}


double NeuralNet::sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}


//Forward propagation gives us the calculated output
void NeuralNet::forwardPropagation()
{
    //initializing z2
    for (int i = 1; i <= n2; ++i) {
		z2[i] = 0.0;
	}

	//initializing z3
    for (int i = 1; i <= n3; ++i) {
		z3[i] = 0.0;
	}

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            z2[j] += vectorImg[i] * w1[i][j];
		}
	}

    for (int i = 1; i <= n2; ++i) {
		a2[i] = sigmoid(z2[i]); // a2 = g(z2)
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            z3[j] += a2[i] * w2[i][j];
		}
	}

    for (int i = 1; i <= n3; ++i) {
		a3[i] = sigmoid(z3[i]);  //a3 = g(z3)
	}
}


// this function calculates the error of the output with respect to our expectedOutput output
double NeuralNet::squaredError()
{
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (a3[i] - expectedOutput[i]) * (a3[i] - expectedOutput[i]);
	}
    res *= 0.5;
    return res;
}
