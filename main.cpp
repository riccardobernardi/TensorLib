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
    for (auto i :cummult({2, 2, 2})) {
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
    // a.print_width();
    // a.print_strides();
}

void test_slice_values00_comples_printed(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    Tensor<int> b = a.slice(0,0);
    // b.print_privates();
}

void test_slice_access(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    Tensor<int> b = a.slice(0,0);
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
    // cout << "anche qui un operazione che crea errori1111" << endl;
    Tensor<int> b = a.flatten(0);
    // cout << "controlliamo il flat del tensore dopo il flattening: " << b._flattened_dim << endl;
    // cout << "anche qui un operazione che crea errori22222" << endl;

    // b.print_privates();

    int d = a({0,0,0,1});
    cout << "il mio valore  di controllo è: " << d << endl;

    // cout << "sto per fare una difficile operazione" << endl;

    int c = b({0,0,1});
    cout << "il mio valore è: " << c << endl;
}

void test_flattening_complex(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    Tensor<int> b = a.flatten(0);

    // 18*1 + 6*2 + 3*1 + 1*0 = 33
    cout << "il mio valore  di controllo è: " << a({1,2,1,0}) << endl;

    // cout << "sto per fare una difficile operazione" << endl;

    // 6*5 + 3*1 + 1*0 = 33
    int c = b({5,1,0});
    cout << "il mio valore è: " << c << endl;
}

void test_check_consistent_initialization(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    // a.initialize({0});
}

void test_check_consistent_initialization_with_permitted_reinit(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    a.initialize({15550,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
}

void test_flattening_complex_quasi_full(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    // cout << "anche qui un operazione che crea errori1111" << endl;
    Tensor<int> b = a.multiFlatten(0,2);
    // cout << "anche qui un operazione che crea errori22222" << endl;

    // b.print_privates();

    int d = a({1,2,1,0});
    cout << "il mio valore  di controllo è: " << d << endl;

    // cout << "sto per fare una difficile operazione" << endl;

    int c = b({11,0});
    cout << "il mio valore è: " << c << endl;
}

void test_flattening_complex_full(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    // cout << "anche qui un operazione che crea errori1111" << endl;
    Tensor<int> b = a.multiFlatten(0,3);
    // cout << "anche qui un operazione che crea errori22222" << endl;

    // b.print_privates();

    int d = a({1,2,1,0});
    cout << "il mio valore  di controllo è: " << d << endl;

    // cout << "sto per fare una difficile operazione" << endl;

    int c = b({33});
    cout << "il mio valore è: " << c << endl;
}

void test_iterations(){
    Tensor<int> a = Tensor<int>({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    /*for(auto i: a){
        cout << i << endl;
    }*/
}

void test_Tensor_constructor_despec(){
    Tensor<int, 3> a = Tensor<int, 3>({1,2,3});
}

void test_Tensor_iterator(){
    Tensor<int> a({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    for(auto i : a){
        cout << i << endl;
    }
}

void test_Tensor_copy_forced(){
    Tensor<int> a = Tensor<int>({1,2,3});
    Tensor<int> b = a.copy();
    Tensor<int> c = a;
    a({0,0,0}) = 11111;
    cout << "a[000]" << a({0,0,0}) << endl;
    cout << "b[000]" << b({0,0,0}) << endl;
    cout << "c[000]" << c({0,0,0}) << endl;
}

void test_Tensor_assignment_copy(){
    Tensor<int> a = Tensor<int>({1,2,3});
    Tensor<int> b(a);
}

void test_Tensor_assignment_move(){
    Tensor<int> a = Tensor<int>({1,2,3});
    Tensor<int> b(std::move(a));
}

void test_Tensor_window(){
    Tensor<int> a({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    Tensor<int> b = a.window(1,1,2);
    for(auto i : b){
        cout << i << endl;
    }
}

void test_Tensor_window_slice(){
    Tensor<int> a({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    Tensor<int> b = a.slice(1,2);
    Tensor<int> c = b.window(1,0,1);
    for(auto i : b){
        cout << i << endl;
    }
}

void test_Tensor_window_slice_10(){
    Tensor<int> a({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    Tensor<int> b = a.slice(1,0);
    Tensor<int> c = b.window(1,0,0);
    for(auto i : c){
        cout << i << endl;
    }
}

void test_Tensor_iteration_fixed(){
    Tensor<int> a({2,3,2,3});
    a.initialize({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35});
    for(auto i = a.begin({0,0,0,0}, 1); i < a.end({0,0,0,0}, 1); ++i){
        cout << (*i) << endl;
    }
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
    t.add(test_flattening_complex,"test_flattening_complex");
    t.add(test_check_consistent_initialization, "test_check_consistent_initialization");
    t.add(test_flattening_complex_quasi_full,"test_flattening_complex_quasi_full");
    t.add(test_flattening_complex_full,"test_flattening_complex_full");
    t.add(test_iterations,"test_iterations");
    t.add(test_check_consistent_initialization_with_permitted_reinit,"test_check_consistent_initialization_with_permitted_reinit");
    t.add(test_Tensor_constructor_despec,"test_Tensor_constructor_despec");
    t.add(test_Tensor_iterator,"test_Tensor_iterator");
    t.add(test_Tensor_copy_forced,"test_Tensor_copy_forced");
    t.add(test_Tensor_assignment_copy,"test_Tensor_assignment_copy");
    t.add(test_Tensor_assignment_move,"test_Tensor_assignment_move");
    t.add(test_Tensor_window,"test_Tensor_window");
    t.add(test_Tensor_window_slice,"test_Tensor_window_slice");
    t.add(test_Tensor_window_slice_10,"test_Tensor_window_slice_10");
    t.add(test_Tensor_iteration_fixed,"test_Tensor_iteration_fixed");
    t.launch_test(-1);
    //t.launch_test(13);

};
