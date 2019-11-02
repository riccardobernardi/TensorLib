#include <iostream>
#include "Tensor.h"

using namespace std;
int main() {

    // Tensor<int> a({1,2,3});

    int x = 2;
    int& y = x;
    int&& tmp = 3;

    cout << x << endl;
    cout << (y) << endl;
    cout << (tmp) << endl;

    x = 0;
    y = 1;
    tmp = 5;

    cout << x << endl;
    cout << (y) << endl;
    cout << (tmp) << endl;

}
