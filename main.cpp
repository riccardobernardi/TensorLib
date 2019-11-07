#include <iostream>
#include "Tensor.h"

using namespace std;

static void test_base(){
    auto NAME = "test_base";
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
    auto NAME = "test_cummult";
    for (auto i :cummult<int>({1, 2, 3})) {
        cout << i << " ";
    }
}

void test_default_constructor(){
    auto NAME = "test_default_constructor";
    //Tensor<> a = Tensor<>();
}

void test_Tensor_constructor_no_compile_hint(){
    auto NAME = "test_Tensor_constructor_no_compile_hint";
    Tensor<int> a = Tensor<int>({1,2,3});
}

void test_Tensor_constructor_with_compile_hint(){
    auto NAME = "test_Tensor_constructor_with_compile_hint";
    Tensor<int,3> a = Tensor<int,3>({1,2,3});
}

void test_Tensor_constructor_with_compile_hint_2(){
    auto NAME = "test_Tensor_constructor_with_compile_hint";
    Tensor<int,3> a = Tensor<int,3>({1,2,3});
}

class Tests{
private:
    vector<function<void()>> _functions;
    vector<string> _names;

public:
    void launch_test(int x){
        if(x == -1){
            for(unsigned long i=0; i<_functions.size();++i){
                try {
                    cout << "--------------------------------------" << endl;
                    _functions[i]();
                    cout<< endl << "concluso test "<< _names[i] <<endl;
                }catch(...) {
                    cout << "Errore al test "<< _names[i] <<endl;
                }
            }
        }else{
            try {
                cout << "--------------------------------------" << endl;
                _functions[x]();
                cout<< endl << "concluso test "<< _names[x] <<endl;
            }catch(...) {
                cout << "Errore al test "<< _names[x] <<endl;
            }
        }

    }
    void add(const function<void()>& a, const string& name){
        _functions.push_back(a);
        _names.push_back(name);
    }
};

int main() {

    Tests t;
    t.add(test_base,"test_base");
    t.add(test_cummult,"test_cummult");
    t.add(test_default_constructor,"test_default_constructor");
    t.add(test_Tensor_constructor_no_compile_hint,"test_Tensor_constructor_no_compile_hint");
    t.add(test_Tensor_constructor_with_compile_hint,"test_Tensor_constructor_with_compile_hint");
    t.launch_test(-1);

}
