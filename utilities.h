//
// Created by rr on 09/11/2019.
//

#ifndef TENSORLIB_UTILITIES_H
#define TENSORLIB_UTILITIES_H

template<class T = size_t>
static std::vector<T> cummult(std::vector<T> a, const size_t& span=0){
    if (a.size() > 0) {
        int size = a.size();
        for (int j = size-1; j > -1; j--){
            if (j < size-1) {
                a[j] *= a[j+1];
            } else {
                a[j] *= span;
            }
        }
    }
    return a;               // TODO vediamo cosa ritornare se il vettore è vuoto
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
