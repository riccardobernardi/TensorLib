#include <iostream>
#include "TensorLib.h"
#include "TestLib.h"

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
    for (auto i :cummult<int>({2, 2, 2})) {
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
    Tensor<int> a = Tensor<int>({1,2,3});
}

void test_Tensor_constructor_with_compile_hint_2(){
    auto NAME = "test_Tensor_constructor_with_compile_hint";
    Tensor<int> a = Tensor<int>({1,2,3});
}

void test_slice_values20(){
    auto NAME = "test_slice_values";
    Tensor<int> a = Tensor<int>({2,2,2});
    a.slice(2,0);
}

void test_slice_values00(){
    auto NAME = "test_slice_values";
    Tensor<int> a = Tensor<int>({2,2,2});
    a.slice(0,0);
}

void test_correct_widths(){
    auto NAME = "test_correct_widths";
    Tensor<int> a = Tensor<int>({2,2,2});
    a.print_width();
    a.print_strides();
}

void test_slice_values00_comples_printed(){
    auto NAME = "test_slice_values";
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
    auto NAME = "test_slice_values";
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    cout << a({0,0,0,1});
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
    t.launch_test(-1);

}
