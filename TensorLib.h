//
// Created by rr on 29/10/2019.
//

#ifndef TENSORLIB_TENSORLIB_H
#define TENSORLIB_TENSORLIB_H

#include <vector>
#include <algorithm>

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

    explicit Tensor<T>(const std::vector<size_t>& a){
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
        _vec = make_shared<T>(a);
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
        cout << "strides " << _strides[index] << endl;
        result._start = index * _strides[index];
        result._stop = result._start + _strides[index];
        result._off = _strides[index] ;
        return result;
    }

    Tensor<T> operator()(initializer_list<size_t> a){

    }

    void print_privates(){
        cout << "private " << "_rank " << _rank << endl;
        cout << "private " << "_size " << _size << endl;
        cout << "private " << "_start " << _start << endl;
        cout << "private " << "_off " << _off << endl;
        cout << "private " << "_stop " << _stop << endl;
    }

private:
    size_t _rank=0;
    size_t _size=0;

    //start della parte lineare del tensore
    size_t _start=0;
    // salto per trovare l altra parte lineare del tensore
    size_t _off=0;
    //stop della parte lineare del tensore
    size_t _stop=0;

    std::shared_ptr<std::vector<T>> _vec;
    std::vector<size_t> _strides;
    std::vector<size_t> _width;
};



#endif //TENSORLIB_TENSORLIB_H
