#include "Tester.h"


Tester::Tester()
{
    //ctor
    testingImageFn = "mnist/t10k-images.idx3-ubyte";
    testingLabelFn = "mnist/t10k-labels.idx1-ubyte";
    modelFn = "model-neural-network.dat";
    reportFn = "testing-report.dat";


    nTesting=10000;
}

Tester::~Tester()
{
    //dtor
    for (int i = 1; i <= n1; ++i)
    {
        delete[] w1[i];
    }

    delete[] vectorImg;

    for (int i = 1; i <= n2; ++i)
    {
        delete[] w2[i];
    }

    delete[] z2;
    delete[] a2;

    delete[] z3;
    delete[] a3;
}


void Tester::memoryAllocation()
{
    // Initializing w1
    for (int i = 1; i <= n1; ++i)
    {
        w1[i] = new double [n2 + 1];
    }

    vectorImg = new double [n1 + 1];

    // Initializing w2
    for (int i = 1; i <= n2; ++i)
    {
        w2[i] = new double [n3 + 1];
    }

    z2 = new double [n2 + 1];
    a2 = new double [n2 + 1];

    z3 = new double [n3 + 1];
    a3 = new double [n3 + 1];
}



void Tester::loadModel(string fileName)
{
    ifstream file(fileName.c_str(), ios::in);

    // Loading w1 from the file
    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1; j <= n2; ++j)
        {
            file >> w1[i][j];
        }
    }

    // Loading w2 from the file
    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            file >> w2[i][j];
        }
    }

    file.close();
}



int Tester::input()
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

    return (int)(number);
}




void Tester::test()
{
    report.open(reportFn.c_str(), ios::out);
    image.open(testingImageFn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(testingLabelFn.c_str(), ios::in | ios::binary ); // Binary label file

    // Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i)
    {
        image.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i)
    {
        label.read(&number, sizeof(char));
    }

    // Neural Network Initialization
    memoryAllocation(); // Memory allocation
    loadModel(modelFn); // Load model (weight matrices) of a trained Neural Network

    int nCorrect = 0;
    for (int sample = 1; sample <= nTesting; ++sample)
    {
        cout << "Sample " << sample << endl;

        // Getting (image, label)
        int label = input();

        forwardPropagation();

        // Prediction by finding the highest value in the output array
        int predict = 1;
        for (int i = 2; i <= n3; ++i)
        {
            if (a3[i] > a3[predict])
            {
                predict = i;
            }
        }
        --predict;

        // Classification result and the squared error
        double error = squaredError();
        printf("Error: %0.6lf\n", error);

        if (label == predict)
        {
            ++nCorrect;
            cout << "Classification: YES. Label = " << label << ". Predict = " << predict << endl << endl;
            report << "Sample " << sample << ": YES. Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
        }
        else
        {
            cout << "Classification: NO.  Label = " << label << ". Predict = " << predict << endl;
            cout << "Image:" << endl;
            for (int j = 1; j <= height; ++j)
            {
                for (int i = 1; i <= width; ++i)
                {
                    cout << img[i][j];
                }
                cout << endl;
            }
            cout << endl;
            report << "Sample " << sample << ": NO.  Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
        }
    }

    // Summary
    double accuracy = (double)(nCorrect) / nTesting * 100.0;
    cout << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    printf("Accuracy: %0.2lf\n", accuracy);

    report << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    report << "Accuracy: " << accuracy << endl;

    report.close();
    image.close();
    label.close();
}
