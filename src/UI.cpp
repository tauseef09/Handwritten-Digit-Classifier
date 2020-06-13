#include "UI.h"

UI::UI()
{
    //ctor
}

UI::~UI()
{
    //dtor
}

// This function takes the user input image matrix as the sample and predicts the number
void UI::prediction(int pic[29][29])
{
    memoryAllocation();
    loadModel(modelFn);


    for (int j = 1; j <= height; ++j)
    {
        for (int i = 1; i <= width; ++i)
        {
            int pos = i + (j - 1) * width;
            vectorImg[pos] = pic[i][j];
        }
    }


    forwardPropagation();
    int predict = 1;
        for (int i = 2; i <= n3; ++i)
        {
            if (a3[i] > a3[predict])
            {
                predict = i;
            }
        }
        --predict;


        cout<< "The number in the image is: " << predict << endl;
}
