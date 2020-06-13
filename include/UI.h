#ifndef UI_H
#define UI_H

#include <Tester.h>
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


class UI : public Tester
{
    public:
        UI();
        virtual ~UI();
        void prediction(int d[29][29]);


    protected:

    private:
};

#endif // UI_H
