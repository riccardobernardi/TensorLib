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
        /*for(auto i : _strides){
            cout << "mie strides:" << i << endl;
        }*/
        _rank = a.size();
        _size = cummult(_width)[0];
    }

    explicit Tensor<T>(const std::vector<size_t> a){
        if (rank!=0){
            // assert(a.size()==rank);
        }
        _width = a;
        _strides = cummult(_width,1);
        _rank = a.size();
        _size = cummult(_width)[0];
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

    void print_privates(){
        cout << "private " << "_rank " << _rank << endl;
        cout << "private " << "_size " << _size << endl;
        cout << "private " << "size del _default " << _default.size() << endl;
        for(auto i : _default){
            cout << "_default is " << get<0>(i) << " " << get<1>(i) << endl;
        }
    }

    size_t operator()(initializer_list<size_t> a){
        std::vector<size_t> b=a;
        for(auto i: _default){
            b.insert(b.begin() + get<0>(i), get<1>(i));
        }
        return ((*_vec)[sum(mult(_strides, b))]);
    }

    Tensor<T> slice(size_t index, size_t value){
        // have to check if there's another one index equal to mine, in this case i've to increase mine
        // why do i have to increase and not to decrease? Because im working in a subset and for example if i take out the second and again the second then the third(the second second) is the third!
        if (_default.size() != 0){
            for(auto j : _default){
                for(auto i: _default){
                    if(index == get<0>(i)){
                        i++;
                    }
                }
            }
        }
        std::vector<size_t> new_width = erase(_width, index);
        Tensor<T> result = Tensor<T>(new_width);
        result._default.push_back(tuple<size_t, size_t>(index, value));
        result._strides = _strides;
        result._vec = _vec;
        return result;
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
