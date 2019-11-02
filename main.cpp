#include <iostream>
#include "Tensor.h"

using namespace std;






class Tests{
public:
    void test_base(){
        auto NAME = "test_base";
        try {
            cout << "--------------------------------------" << endl;
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
        }catch(...) {
            cout << "Errore al test" << "test_base";
        }
    }

    void test_cummult(){
        auto NAME = "test_cummult";
        try {
            cout << "--------------------------------------" << endl;
            for (auto i :cummult<int>({1, 2, 3})) {
                cout << i << " ";
            }
        }catch(...){
            cout << "Errore al test" << NAME;
        }
    }

    void test_Tensor_constructor_no_compile_hint(){
        auto NAME = "test_cummult";
        try {
            Tensor<int> a = Tensor<int>({1,2,3});
        }catch(...){
            cout << "Errore al test" << NAME;
        }
    }

    void test_Tensor_constructor_with_compile_hint(){
        auto NAME = "test_cummult";
        try {
            Tensor<int,3> a = Tensor<int,3>({1,2,3});
        }catch(...){
            cout << "Errore al test" << NAME;
        }
    }
};

int main() {

    // Tensor<int> a({1,2,3});

    Tests a = Tests();
    a.test_base();
    a.test_cummult();

}
