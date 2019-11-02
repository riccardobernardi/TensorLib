#include <iostream>
#include "Tensor.h"

using namespace std;






class Tests{
public:
    void test_base(){
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

    void test_cummult(){
        for (auto i :cummult<int>({1,2,3})){
            cout<<i<<" ";
        }
    }
};

int main() {

    // Tensor<int> a({1,2,3});

    Tests a = Tests();
    a.test_base();
    a.test_cummult();

}
