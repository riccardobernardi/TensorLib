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
    Tensor<int,3> a = Tensor<int,3>({1,2,3});
}

void test_Tensor_constructor_with_compile_hint_2(){
    auto NAME = "test_Tensor_constructor_with_compile_hint";
    Tensor<int,3> a = Tensor<int,3>({1,2,3});
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
    t.launch_test(-1);

}
