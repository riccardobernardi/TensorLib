//
// Created by rr on 09/11/2019.
//

#include "TensorLib.h"

#ifndef TENSORLIB_UTILITIES_H
#define TENSORLIB_UTILITIES_H

static std::vector<size_t> cummult(std::vector<size_t> a){
    // std::reverse(a.begin(),a.end());

    std::vector<size_t> b;
    int tmp = 1;
    for(int j=a.size()-1; j>-1;--j){
        b.push_back(tmp);
        tmp *= a[j];
    }

    reverse(b.begin(),b.end());
    return b;
}

template <class T>
std::vector<T> erase(std::vector<T> a, T index){
    std::vector<size_t> b;
    for(int i = 0; i< a.size(); ++i){
        if (i!=index){
            b.push_back(a[i]);
        }
    }
    return b;
}

template <class T>
std::vector<T> mult(std::vector<T> a, std::vector<T> b){
    assert(a.size() == b.size());
    std::vector<T> c;
    for(int i =0; i< a.size(); ++i){
        c.push_back(a[i]*b[i]);
    }
    return c;
}

template <class T>
T mult(std::vector<T> a){
    T tmp = 1;
    for(auto i : a){
        tmp *= i;
    }
    return tmp;
}

template <class T>
T sum(std::vector<T> a){
    T tmp=0;
    for(auto i : a){
        tmp += i;
    }
    return tmp;
}

#endif //TENSORLIB_UTILITIES_H