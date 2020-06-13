#include "Trainer.h"

Trainer::Trainer()
{
    //ctor

    //storing the required file names in the strings
    trainingImageFn = "mnist/train-images.idx3-ubyte";
    trainingLabelFn = "mnist/train-labels.idx1-ubyte";
    modelFn = "model-neural-network.dat";
    reportFn = "training-report.dat";


    nTraining = 60000;
    epochs = 512;
    learningRate = 1e-3;
    momentum = 0.9;
    epsilon = 1e-3;
}

Trainer::~Trainer()
{
    //dtor
}


void Trainer::abstract()
{
    //overriding the pure virtual function
}


void Trainer::memoryAllocation()
{
    // Allocating memory for the weight and delta matrix for the input to hidden layer
    for (int i = 1; i <= n1; ++i)
    {
        w1[i] = new double [n2 + 1];
        deltaW1[i] = new double [n2 + 1];
    }

    vectorImg = new double [n1 + 1];

    // Allocating memory for the weight and delta matrix for the hidden to output layer
    for (int i = 1; i <= n2; ++i)
    {
        w2[i] = new double [n3 + 1];
        deltaW2[i] = new double [n3 + 1];
    }

    z2 = new double [n2 + 1];
    a2 = new double [n2 + 1];
    smallDel2 = new double [n2 + 1];

    z3 = new double [n3 + 1];
    a3 = new double [n3 + 1];
    smallDel3 = new double [n3 + 1];

    // Randomly initializing w1
    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1; j <= n2; ++j)
        {
            int sign = rand() % 2;

            w1[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1)
            {
                w1[i][j] = - w1[i][j];
            }
        }
    }

    // Randomly initializing w2
    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            int sign = rand() % 2;

            w2[i][j] = (double)(rand() % 10 + 1) / (10.0 * n3);
            if (sign == 1)
            {
                w2[i][j] = - w2[i][j];
            }
        }
    }
}


// Backpropagation fixes the values of the weights of the previous layer depending on the amount of error found in one layer
// Reference to the formulas used - https://publications.idiap.ch/downloads/reports/1995/95-04.pdf?fbclid=IwAR1SAXs7e3uTVSI8o0Xt3Qd3BJ9HfOMD4Wn4osXphKPYutpXMOh9ptgkR78
void Trainer::backPropagation()
{
    double sum;

    for (int i = 1; i <= n3; ++i)
    {
        smallDel3[i] = a3[i] * (1 - a3[i]) * (expectedOutput[i] - a3[i]);
    }

    for (int i = 1; i <= n2; ++i)
    {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j)
        {
            sum += w2[i][j] * smallDel3[j];
        }
        smallDel2[i] = a2[i] * (1 - a2[i]) * sum;
    }

    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            deltaW2[i][j] = (learningRate * smallDel3[j] * a2[i]) + (momentum * deltaW2[i][j]);
            w2[i][j] += deltaW2[i][j];
        }
    }

    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1 ; j <= n2 ; j++ )
        {
            deltaW1[i][j] = (learningRate * smallDel2[j] * vectorImg[i]) + (momentum * deltaW1[i][j]);
            w1[i][j] += deltaW1[i][j];
        }
    }
}


// Training process runs the FP and BP for every training example until an error less than epsilon is found
// If error does not go less than epsilon, it runs the process max epoch times
// The function returns the number of iterations needed to minimize the error
int Trainer::learningProcess()
{

    // Initializing
    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1; j <= n2; ++j)
        {
            deltaW1[i][j] = 0.0;
        }
    }

    //Initializing
    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            deltaW2[i][j] = 0.0;
        }
    }

    for (int i = 1; i <= epochs; ++i)
    {
        forwardPropagation();
        backPropagation();
        if (squaredError() < epsilon)
        {
            return i;
        }
    }
    return epochs;
}



void Trainer::input()
{
    // Reading image
    char number;
    for (int j = 1; j <= height; ++j)
    {
        for (int i = 1; i <= width; ++i)
        {
            image.read(&number, sizeof(char));
            if (number == 0)
            {
                img[i][j] = 0;
            }
            else
            {
                img[i][j] = 1;
            }
        }
    }

    cout << "Image:" << endl;
    for (int j = 1; j <= height; ++j)
    {
        for (int i = 1; i <= width; ++i)
        {
            cout << img[i][j];
        }
        cout << endl;
    }

    for (int j = 1; j <= height; ++j)
    {
        for (int i = 1; i <= width; ++i)
        {
            int pos = i + (j - 1) * width;
            vectorImg[pos] = img[i][j];
        }
    }

    // Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i)
    {
        expectedOutput[i] = 0.0;
    }
    expectedOutput[number + 1] = 1.0;

    cout << "Label: " << (int)(number) << endl;
}




void Trainer::writeMatrix(string fileName)
{
    ofstream file(fileName.c_str(), ios::out);

    // Saving w1 to the file
    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1; j <= n2; ++j)
        {
            file << w1[i][j] << " ";
        }
        file << endl;
    }

    // Saving w2 to the file
    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            file << w2[i][j] << " ";
        }
        file << endl;
    }

    file.close();
}



void Trainer::train()
{
    report.open(reportFn.c_str(), ios::out);
    image.open(trainingImageFn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(trainingLabelFn.c_str(), ios::in | ios::binary ); // Binary label file

	// Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}

	// Neural Network Initialization
    memoryAllocation();

    for (int sample = 1; sample <= nTraining; ++sample) {
        cout << "Sample " << sample << endl;

        // Getting (image, label)
        input();

		// Learning process: Perceptron (Forward procedure) - Back propagation
        int nIterations = learningProcess();

		// Write down the squared error
		cout << "No. iterations: " << nIterations << endl;
        printf("Error: %0.6lf\n\n", squaredError());
        report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << squaredError() << endl;

		// Save the current network (weights)
		if (sample % 100 == 0) {
			cout << "Saving the network to " << modelFn << " file." << endl;
			writeMatrix(modelFn);
		}
    }

	// Save the final network
    writeMatrix(modelFn);

    report.close();
    image.close();
    label.close();
}
