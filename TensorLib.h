//
// Created by rr on 29/10/2019.
//

#ifndef TENSORLIB_TENSORLIB_H
#define TENSORLIB_TENSORLIB_H

#include <vector>

using namespace std;

template<class T>
static std::vector<T> cummult(std::vector<T> a){
    std::vector<T> b;
    for(int j=0; j<a.size();++j){
        int tmp = 1;
        for(int i=0; i< a.size()-j; ++i){
            tmp = tmp*a[i+j];
        }
        b.push_back(tmp);
    }
    b[b.size()-1] = 1;


    return b;
}

template<class T = int, size_t rank=0>
class Tensor {
public:
    // when the rank is not specified
    Tensor<T,0>(std::initializer_list<size_t>&& a){
        if (rank!=0){
            assert(a.size()==rank);
        }
        _width = a;
        _strides = cummult(_width);
        _rank = a.size();
        _size = cummult(_width)[_width.size()-1];
        cout << _size << endl;
        cout << to_string(_rank) << endl;
    }

    // initialize with an array that will be represented as a tensor
    void initialize(std::initializer_list<T>&& a){
        assert(a.size() == _size);
    }

    void slice(size_t start, size_t increment){
        std::vector<size_t> b;
        b.push_back(start*_width[start]);
        cout << start << endl;
        cout << _width[start] << endl;

        for(int i=0; i<((cummult(_width)[_width.size()-1])/start); ++i){
            cout << " halo ";
            size_t index = start*_width[start] + int((start*_width[start]==0?1:start*_width[start]) / (_size)/start);
            cout << start*_width[start] + int((start*_width[start]==0?1:start*_width[start]) / (_size)/start);
            b.push_back(index);
        }
    }

private:
    size_t _rank=0;
    size_t _size=0;
    std::shared_ptr<std::vector<T>> _vec;
    std::vector<size_t> _strides;
    std::vector<size_t> _width;
};



#endif //TENSORLIB_TENSORLIB_H
