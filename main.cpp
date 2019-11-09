#include <iostream>
#include "TensorLib.h"
#include "TestLib.h"

using namespace std;

static void test_base(){
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
    for (auto i :cummult<int>({2, 2, 2})) {
        cout << i << " ";
    }
}

void test_default_constructor(){
    //Tensor<> a = Tensor<>();
}

void test_Tensor_constructor_no_compile_hint(){
    Tensor<int> a = Tensor<int>({1,2,3});
}

void test_Tensor_constructor_with_compile_hint(){
    Tensor<int> a = Tensor<int>({1,2,3});
}

void test_Tensor_constructor_with_compile_hint_2(){
    Tensor<int> a = Tensor<int>({1,2,3});
}

void test_slice_values20(){
    Tensor<int> a = Tensor<int>({2,2,2});
    a.slice(2,0);
}

void test_slice_values00(){
    Tensor<int> a = Tensor<int>({2,2,2});
    a.slice(0,0);
}

void test_correct_widths(){
    Tensor<int> a = Tensor<int>({2,2,2});
    a.print_width();
    a.print_strides();
}

void test_slice_values00_comples_printed(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    Tensor<int> b = a.slice(0,0);
    b.print_privates();
}

void test_slice_access(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    Tensor<int> b = a.slice(0,0);
    // b.print_privates();
    cout << b({0,0,1});
}

void test_slice_access_more_complex(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    Tensor<int> b = a.slice(0,0);

    cout << a({0,2,0,2}) << endl;
    cout << b({2,0,2}) << endl;
}

void test_access(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    cout << a({0,0,0,1});
}

/*void test_torsello1(){
    std::string s1("Hello");
    std::string s2(" ");
    std::string s3("World!\n");
    const std::string &s = s1 + s2 + s3; std::cout << s; // OK?
}*/

/*class test_torsello2c{
    static const std::string &getString() {
        return std::string ("Hello");
    }
public:
    void func () {
        std::cout << getString(); // OK?
    }
};

void test_torsello2(){
    try {
        test_torsello2c().func();
    }catch(...){
        cout << "failed because object returned is temporary and do it's deleted as returns";
    }
}*/

void test_flattening(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    Tensor<int> b = a.flatten(0,1);
    cout << "vvvvvvvvvvvvvvvvvvvvvv" << endl;
    b.print_privates();
    cout << "^^^^^^^^^^^^^^^^^^^^^^" << endl;

    cout << "il mio valore è: " << b({5,0,1});
}

int main() {

    Test t;
    t.add(test_base,"test_base");
    t.add(test_cummult,"test_cummult");
    t.add(test_default_constructor,"test_default_constructor");
    t.add(test_Tensor_constructor_no_compile_hint,"test_Tensor_constructor_no_compile_hint");
    t.add(test_Tensor_constructor_with_compile_hint,"test_Tensor_constructor_with_compile_hint");
    t.add(test_slice_values20,"test_slice_values20" );
    t.add(test_slice_values00,"test_slice_values00" );
    t.add(test_correct_widths, "test_correct_widths");
    t.add(test_slice_values00_comples_printed,"test_slice_values00_comples_printed");
    t.add(test_access,"test_access");
    t.add(test_slice_access,"test_slice_access");
    t.add(test_slice_access_more_complex,"test_slice_access_more_complex");
    // t.add(test_torsello1, "test_torsello1");
    // t.add(test_torsello2,"test_torsello2"); // bloccato perchè genera errore giustamente
    t.add(test_flattening,"test_flattening");
    t.launch_test(-1);

}
