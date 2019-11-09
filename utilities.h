//
// Created by rr on 09/11/2019.
//

#ifndef TENSORLIB_UTILITIES_H
#define TENSORLIB_UTILITIES_H

template<class T = size_t>
static std::vector<T> cummult(std::vector<T> a, T span=0){
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
T sum(std::vector<T> a){
    T tmp=0;
    for(auto i : a){
        tmp += i;
    }
    return tmp;
}

#endif //TENSORLIB_UTILITIES_H
