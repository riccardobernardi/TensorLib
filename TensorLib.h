//
// Created by rr on 29/10/2019.
//

#ifndef TENSORLIB_TENSORLIB_H
#define TENSORLIB_TENSORLIB_H

#include <vector>
#include <algorithm>
#include <type_traits>

using namespace std;

template<class T>
static std::vector<T> cummult(std::vector<T> a, size_t span=0){
    std::reverse(a.begin(),a.end());

    std::vector<T> b;
    for(int j=0; j<a.size();++j){
        int tmp = 1;
        for(int i=0; i< a.size()-j-span; ++i){
            tmp = tmp*a[i+j];
        }
        b.push_back(tmp);
    }
    b[b.size()-1] = 1;


    return b;
}

std::vector<size_t> erase(std::vector<size_t> a, size_t index){
    std::vector<size_t> b;
    for(int i = 0; i< a.size(); ++i){
        if (i!=index){
            b.push_back(a[i]);
        }
    }
    return b;
}

std::vector<size_t> mult(std::vector<size_t> a, std::vector<size_t> b){
    assert(a.size() == b.size());
    std::vector<size_t> c;
    for(int i =0; i< a.size(); ++i){
        c.push_back(a[i]*b[i]);
    }
    return c;
}

size_t sum(std::vector<size_t> a){
    size_t tmp=0;
    for(auto i : a){
        tmp += i;
    }
    return tmp;
}

template<class T = int, size_t rank=0>
class Tensor {
public:
    // when the rank is not specified
    Tensor<T>(std::initializer_list<size_t>&& a){
        if (rank!=0){
            // assert(a.size()==rank);
        }
        _width = a;
        _strides = cummult(_width,1);
        for(auto i : _strides){
            cout << "mie strides:" << i << endl;
        }
        _rank = a.size();
        _size = cummult(_width)[0];
        cout << "size: " << _size << endl;
        //cout << to_string(_rank) << endl;
    }

    explicit Tensor<T>(const std::vector<size_t> a){
        if (rank!=0){
            // assert(a.size()==rank);
        }
        _width = a;
        _strides = cummult(_width,1);
        _rank = a.size();
        _size = cummult(_width)[0];
        cout << "size: " << _size << endl;
        //cout << to_string(_rank) << endl;
    }

    // initialize with an array that will be represented as a tensor
    void initialize(std::initializer_list<T>&& a){
        assert(a.size() == _size);
        _vec = make_shared<std::vector<T>>(a);
    }

    void print_width(){
        for(auto i : _width){
            cout << i << endl;
        }
        cout << "*******" << endl;
    }

    void print_strides(){
        for(auto i : _strides){
            cout << i << endl;
        }
        cout << "*******" << endl;
    }

    Tensor<T> slice(size_t index, size_t value){
        std::vector<size_t> new_width = erase(_width, value);
        Tensor<T> result = Tensor<T>(new_width);
        //cout << result.print_privates();
        cout << "strides " << _strides[index] << endl;
        result._default.push_back(tuple<size_t, size_t>(index, value));

        return result;
    }

    size_t operator()(initializer_list<size_t> a){
        cout << "vettore prima" << endl;
        std::vector<size_t> b=a;
        for(auto i : b){
            cout << i << " ";
        }
        cout << endl;
        cout << "vettore dopo" << endl;
        cout << "size della _default : " << _default.size() << endl;
        //cout << "tento l inserimento manuale";
        //cout << "valore strano: " << get<0>(_default[0]);
        //b.insert(b.begin() + get<0>(_default[0]), get<1>(_default[0]));
        //cout << "finito inserimento manuale";
        for(auto i: _default){
            cout << "inserisco";
            b.insert(b.begin() + get<0>(i), get<1>(i));
        }
        for(auto i : b){
            cout << i << " ";
        }
        cout << endl;
        cout << "indice puntato : " << sum(mult(_strides, b)) << endl;
        cout << "result " << (*_vec)[sum(mult(_strides, b))] << endl;
        return ((*_vec)[sum(mult(_strides, b))]);
    }

    void print_privates(){
        cout << "private " << "_rank " << _rank << endl;
        cout << "private " << "_size " << _size << endl;
        for(auto i : _default){
            cout << "_default is " << get<0>(i) << " " << get<1>(i);
        }
    }

private:
    size_t _rank=0;
    size_t _size=0;

    // when you do slicing then there is an index that it is defaulted
    std::vector<tuple<size_t, size_t>> _default;

    std::shared_ptr<std::vector<T>> _vec;
    std::vector<size_t> _strides;
    std::vector<size_t> _width;
};



#endif //TENSORLIB_TENSORLIB_H
