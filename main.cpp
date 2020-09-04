#include <iostream>
#include "NeuralNet.h"
#include "Trainer.h"
#include "Tester.h"
#include "UI.h"

//#define TrainTest 1

using namespace std;

int main()
{

    #ifdef TrainTest

    Trainer trainer;
    Tester tester;

    trainer.train();
    tester.test();

    #endif // TrainTest


    // the dimensions of the matrix needs to be constant so that the matrix can later be passed as a parameter
    const int height=28, width=28;
    int pic[width+1][height+1];


    //taking the image as input from the user
    cout<< "The image:\n"<<endl;
    for (int j = 1; j <= height; j++)
    {
        for (int i = 1; i <= width; i++)
        {
            char number;
            cin >> number;
            if(number=='0')
            {
                pic[i][j]=0;

            }
            else
            {
                pic[i][j]=1;
            }

        }

    }

    cout<<endl;

    UI ui;
    ui.prediction(pic); //passing the image matrix as the parameter to the prediction function

    return 0;
}
