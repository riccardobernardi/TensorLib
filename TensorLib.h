//
// Created by rr on 29/10/2019.
//

#ifndef TENSORLIB_TENSORLIB_H
#define TENSORLIB_TENSORLIB_H

#include <vector>
#include <algorithm>
#include <type_traits>
#include "utilities.h"
#include <cstdarg>
#include <math.h>

using namespace std;

//##########################################################################
//              iterators declaration
//##########################################################################

template<class T, size_t rank>
class TensorIterator;

template<class T, size_t rank>
class TensorIteratorFixed;

template<class T, size_t rank>
TensorIterator<T, rank> operator+(const size_t n, const TensorIterator<T, rank>& iter){
    return iter + n;
};

template<class T, size_t rank>
TensorIteratorFixed<T, rank> operator+(const size_t n,  const TensorIteratorFixed<T, rank>& iter){
    return iter + n;
};

//##########################################################################
//              STATIC RANK TENSOR
//##########################################################################

template<class T = size_t, size_t rank=0>
class Tensor {
public:

    // iterators need to access fields of width
    friend class TensorIterator<T, rank>;
    friend class TensorIteratorFixed<T, rank>;

    // in the slice or flatten we need to access to tensor of upper template to modify metadata
    friend class Tensor<T, rank + 1>;

    TensorIterator<T, rank> begin(){
        std::vector<size_t> ind(widths.size(), 0);
        return TensorIterator<T,rank>(*this,ind);
    }

    TensorIterator<T, rank> end(){
        std::vector<int> ind(widths.size(), 0);
        ind.at(0) = widths[0];
        return TensorIterator<T,rank>(*this,ind);
    }

    TensorIteratorFixed<T, rank> begin(const std::vector<int>& starting_indexes, const size_t& sliding_index){
        return TensorIteratorFixed<T,rank>(*this,starting_indexes,sliding_index);
    }

    TensorIteratorFixed<T, rank> end(const std::vector<int>& starting_indexes, const size_t& sliding_index){
        std::vector<int> ind(widths.size());
        ind[sliding_index] = widths[sliding_index];

        return TensorIteratorFixed<T,rank>(*this,ind,sliding_index);
    }

    // data is hard copied, so data is not shared with the older tensor
    Tensor<T, rank> copy() const {
        Tensor<T, rank> a(widths);
        a.strides = strides;
        a.offset = offset;
        a.data = make_shared<std::vector<T>>(*data);
        return a;
    }

    // constructor that takes widths as initializer list
    Tensor<T,rank>(std::initializer_list<size_t> a){
        assert(a.size() == rank);
        for(auto i : a){
            assert(i>0);
        }
        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0);
    }

    // constructor that takes widths as vector
    Tensor<T,rank>(const std::vector<size_t> a){
        assert(a.size() == rank);

        for(auto i : a){
            assert(i>0);
        }
        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0);
    }

    // constructor that takes widths as reference of initializer list
    Tensor<T,rank>(const std::vector<size_t>& a){
        assert(a.size() == rank);

        cout << "sto usando il generico con valore rank:" << rank << endl;

        for(auto i : a){
            assert(i>0);
        }

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); 
    }

    // copy constructor
    Tensor<T, rank>(const Tensor<T, rank>& a) : widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){}

    //move constructor, tensor is passed as parameter and it is emptied of its data
    Tensor<T, rank>(Tensor<T, rank>&& a) : widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){
        //we cannot change "a" metadata because they are immutable
        a.data = std::shared_ptr<std::vector<T>>();
    }

    // constructor that takes widths as initializer list and the relative vector of data
    Tensor<T, rank>(const std::initializer_list<size_t>& a, std::vector<T>& new_data) {
        assert(a.size() == rank && new_data.size() == mult<T>(a));

        for(auto i : a){
            assert(i>0);
        }

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<new_data>;
        offset = 0;
    }

    // initialize with an array that will be represented as a ttensor
    void initialize(const std::initializer_list<T>& a){
        assert ( (a.size() == widths.size()));
        data = make_shared<std::vector<T>>(a);
    }

    // get/set for data, giving back the refence you can also assign
    T& operator()(const initializer_list<size_t>& indices){
        assert(indices.size() == widths.size());
        assert(data);

        std::vector<size_t> indices_v = indices;
        size_t tmp = 0;

        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            tmp += indices_v[i] * strides[i];
        }
        tmp += offset;

        return ((*data)[tmp]);
    }

    // get/set for data, giving back the refence you can also assign
    T& operator()(vector<int> indices_v){
        assert(indices_v.size() == widths.size());
        assert(data);
        size_t tmp = 0;

        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            tmp += indices_v[i] * strides[i];
        }
        tmp += offset;

        return (*data)[tmp];
    }

    //allow to set an element
    void set(const initializer_list<size_t> indices, const T& value){
        assert(indices.size() == widths.size());
        assert(data);

        std::vector<size_t> indices_v = indices;
        size_t tmp = 0;
        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            tmp += indices_v[i] * strides[i];
        }

        tmp += offset;

        (*data)[tmp] = value;
    }

    // allow to get an element
    T get(const initializer_list<size_t> indices) const {
        assert(indices.size() == widths.size());
        assert(data);

        std::vector<size_t> indices_v = indices;
        size_t tmp = 0;
        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            tmp += indices_v[i] * strides[i];
        }

        tmp += offset;

        return (*data)[tmp];
    }

    // does the slice on the dimension on the position index on the value value
    Tensor<T, rank - 1> slice(const size_t&  index, const size_t& value){
        assert(index >= 0 && index < widths.size());
        assert(value >= 0 && value < widths[index]);
        assert(data);

        Tensor<T, rank - 1> a = Tensor<T, rank - 1>();
        a.widths = erase<size_t>(widths, index);
        a.strides = erase<size_t>(strides, index);
        a.offset += (strides[index] * value);
        a.data = data;


        return a;
    }

    // single flatten, takes left dimension index, es: dim = [2,3,5,1], flatten(2) takes dimensions that are large 5 and 1
    Tensor<T, rank - 1> flatten(const size_t& start){
        assert(start >= 0 && start < widths.size()-1);
        assert(data);
        std::vector<size_t> new_width;
        size_t tmp=1;
        size_t stop = start + 1;
        for(int i=0; i<widths.size(); ++i){
            if (i<start || i>stop) {
                new_width.push_back(widths[i]);
            } else {
                tmp*=widths[i];
                if(i==stop){
                    new_width.push_back(tmp);
                }
            }
        }

        Tensor<T, rank - 1> a = Tensor<T, rank - 1>(new_width);

        a.strides = cummult(new_width);
        a.data = data;
        a.offset = offset;
        return a;
    }

    // multiple flatten, do the same thing as before but there is no constraint about how many dimensions to flattem at the same time, dimensions are all consecutives
    Tensor<T> multiFlatten(const size_t& start, const size_t& stop){
        assert(stop >= start);
        assert(start >= 0 && start < widths.size());
        assert(stop >= 0 && stop < widths.size());
        assert(       ( widths.size() - (stop - start) ) > 0      );
        
        assert(data);

        std::vector<size_t> new_width;
        size_t tmp=1;



        for(size_t i=0;i<widths.size();++i){
            if (i<start || i>stop) {
                new_width.push_back(widths[i]);
            } else {
                tmp*=widths[i];
                if(i==stop){
                    new_width.push_back(tmp);
                }
            }
        }

        Tensor<T> a = Tensor<T>(new_width);

        a.strides = cummult(new_width);
        a.data = data;
        a.offset = offset;

        return a;
    }

    // window, takes index of dimension to be shrinked and the ends that will be accepted after operation.
    Tensor<T, rank> window(const size_t& index, const size_t& start, const size_t& stop){
        assert(stop > start);
        assert(widths[index] > stop);
        assert(start >= 0);
        assert(data);

        Tensor<T, rank> a = Tensor<T, rank>(widths);

        a.widths[index] = stop - start + 1;
        a.offset += a.strides[index] * start;
        a.data = data;
        a.strides = strides;


        return a;
    }

private:

    // metadata : immutable
    std::vector<size_t> widths;
    std::vector<size_t> strides;
    size_t offset=0;

    //data : mutable
    std::shared_ptr<std::vector<T>> data;

    //default constructor, we use it in slice/windows to build a clean rank-1 tensor 
    Tensor<T, rank>() : widths(), strides(), data(), offset() {}

};

//##########################################################################
//              DYNAMIC RANK TENSOR
//##########################################################################

template<class T>
class Tensor<T,0> {
public:
    // iterators need to access fields of width
    friend class TensorIterator<T, 0>;
    friend class TensorIteratorFixed<T, 0>;

    // in the slice or flatten we need to access to static tensor template to modify metadata
    template<typename S, size_t rank> friend class Tensor;


    TensorIterator<T, 0> begin(){
        std::vector<int> ind(widths.size(), 0);
        return TensorIterator<T,0>(*this,ind);
    }

    TensorIterator<T, 0> end(){
        std::vector<int> ind(widths.size(), 0);
        ind.at(0) = widths[0];
        return TensorIterator<T,0>(*this,ind);
    }

    TensorIteratorFixed<T, 0> begin(const std::vector<int>& starting_indexes, const size_t& sliding_index){
        return TensorIteratorFixed<T, 0>(*this,starting_indexes,sliding_index);
    }

    TensorIteratorFixed<T, 0> end(const std::vector<int>& starting_indexes, const size_t& sliding_index){
        std::vector<int> ind = std::vector<int>(widths.size());
        ind[sliding_index] = widths[sliding_index];

        return TensorIteratorFixed<T, 0>(*this,ind,sliding_index);
    }

    // data is hard copied, so data is not shared with the older tensor
    Tensor<T> copy() const {
        Tensor<T, 0> a(widths);
        a.strides = strides;
        a.offset = offset;
        a.data = make_shared<std::vector<T>>(*data);
        return a;
    };

    // constructor that takes widths as initializer list
    Tensor<T>(std::initializer_list<size_t> a){
        assert(a.size() > 0);
        for(auto i : a){
            assert(i>0);
        }

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0);
    }

    // constructor that takes widths as vector
    Tensor<T>(std::vector<size_t> a){
        for(auto i : a){
            assert(i>0);
        }
        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0);
    }

    // copy constructor
    Tensor<T>(Tensor<T>& a): widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){}

    //move constructor, tensor is passed as parameter and it is emptied of its data
    Tensor<T>(Tensor<T>&& a) : widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){
        //we cannot change "a" metadata because they are immutable
        a.data = std::shared_ptr<std::vector<T>>();
    }

    // constructor that takes widths as initializer list and the relative vector of data
    Tensor<T>(std::initializer_list<size_t>& a, std::vector<T>& new_data) {
        assert(a.size() > 0 && new_data.size() == mult<T>(a));
        for(auto i : a){
            assert(i>0);
        }

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<new_data>;
        offset = 0;
    }

    // initialize with an array that will be represented as a ttensor
    void initialize(const std::initializer_list<T>& a){
        assert ( (a.size() == (*data).size()));
        data = make_shared<std::vector<T>>(a);
    }

    // get/set for data, giving back the refence you can also assign
    T& operator()(const initializer_list<size_t>& indices){
        assert(indices.size() == widths.size());
        assert(data);

        std::vector<size_t> indices_v = indices;
        size_t tmp = 0;

        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            tmp += indices_v[i] * strides[i];
        }
        tmp += offset;

        return (*data)[tmp];
    }

    // get/set for data, giving back the refence you can also assign
    T& operator()(vector<int> indices_v){
        assert(indices_v.size() == widths.size());
        assert(data);

        size_t tmp = 0;

        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            tmp += indices_v[i] * strides[i];
        }
        tmp += offset;

        return (*data)[tmp];
    }

    //allow to set an element
    void set(const initializer_list<size_t> indices, const T& value){
        assert(indices.size() == widths.size());
        assert(data);

        std::vector<size_t> indices_v = indices;
        size_t tmp = 0;
        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            tmp += indices_v[i] * strides[i];
        }

        tmp += offset;

        (*data)[tmp] = value;
    }

    // allow to get an element
    T get(const initializer_list<size_t> indices) const {
        assert(indices.size() == widths.size());
        assert(data);

        std::vector<size_t> indices_v = indices;
        size_t tmp = 0;
        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            tmp += indices_v[i] * strides[i];
        }

        tmp += offset;

        return (*data)[tmp];
    }

    // does the slice on the dimension on the position index on the value value
    Tensor<T> slice(const size_t&  index, const size_t& value){
        assert(index >= 0 && index < widths.size());
        assert(value >= 0 && value < widths[index]);
        assert(widths.size() > 1);
        assert(data);

        Tensor<T> a = Tensor<T>(widths);
        a.widths = erase<size_t>(widths, index);
        a.strides = erase<size_t>(strides, index);
        a.offset += (strides[index] * value);
        a.data = data;

        return a;
    }

    // single flatten, takes left dimension index, es: dim = [2,3,5,1], flatten(2) takes dimensions that are large 5 and 1
    Tensor<T> flatten(const size_t& start){
        assert(start >= 0 && start < widths.size() - 1);
        assert(widths.size() >= 2);
        assert(data);

        std::vector<size_t> new_width;
        size_t tmp=1;
        size_t stop = start + 1;


        for(int i=0;i<widths.size();++i){
            if (i<start || i>stop) {
                new_width.push_back(widths[i]);
            } else {
                tmp*=widths[i];
                if(i==stop){
                    new_width.push_back(tmp);
                }
            }
        }

        Tensor<T> a = Tensor<T>(new_width); 

        a.strides = cummult(new_width);
        a.data = data;
        a.offset = offset;


        return a;
    }

    // multiple flatten, do the same thing as before but there is no constraint about how many dimensions to flattem at the same time, dimensions are all consecutives
    Tensor<T> multiFlatten(const size_t& start, const size_t& stop){
        assert(stop >= start);        
        assert(start >= 0 && start < widths.size());
        assert(stop >= 0 && stop < widths.size());
        assert(widths.size() >= 2);
        assert(       ( widths.size() - (stop - start) ) > 0      );

        assert(data);

        std::vector<size_t> new_width;
        size_t tmp=1;


        for(size_t i=0;i<widths.size();++i){
            if (i<start || i>stop) {
                new_width.push_back(widths[i]);
            } else {
                tmp*=widths[i];
                if(i==stop){
                    new_width.push_back(tmp);
                }
            }
        }

        Tensor<T> a = Tensor<T>(new_width);

        a.strides = cummult(new_width);
        a.data = data;
        a.offset = offset;
        return a;
    }

    // window, takes index of dimension to be shrinked and the ends that will be accepted after operation.
    Tensor<T> window(const size_t& index, const size_t& start, const size_t& stop){
        assert(stop >= start);
        assert(widths[index] > stop);
        assert(start >= 0);
        assert(data);

        Tensor<T> a = Tensor<T>(widths);

        a.widths[index] = stop - start + 1;
        a.offset += a.strides[index] * start;
        a.data = data;
        a.strides = strides;

        return a;
    }

private:

    // metadata : immutable
    std::vector<size_t> widths;
    std::vector<size_t> strides;
    size_t offset=0;

    //data : mutable
    std::shared_ptr<std::vector<T>> data;
};

//##########################################################################
//              STATIC RANK 1 TENSOR
//##########################################################################

template<class T>
class Tensor<T,1> {
public:
    // iterators need to access fields of width
    friend class TensorIterator<T, 1>;
    friend class TensorIteratorFixed<T, 1>;

    // in the slice or flatten we need to access to tensor of upper template to modify metadata
    friend class Tensor<T, 2>;

    TensorIterator<T, 1> begin(){
        std::vector<int> ind(1, 0);
        return TensorIterator<T, 1>(*this,ind);
    }

    TensorIterator<T, 1> end(){
        std::vector<int> ind(1, 0);
        ind.at(0) = widths[0];
        return TensorIterator<T, 1>(*this,ind);
    }

    TensorIteratorFixed<T, 1> begin(const std::vector<int>& starting_indexes, const size_t& sliding_index){
        return TensorIteratorFixed<T, 1>(*this,starting_indexes,sliding_index);
    }

    TensorIteratorFixed<T, 1> end(const std::vector<int>& starting_indexes, const size_t& sliding_index){
        std::vector<int> ind(1);
        ind[0] = widths[0];

        return TensorIteratorFixed<T, 1>(*this,ind,sliding_index);
    }

    // data is hard copied, so data is not shared with the older tensor
    Tensor<T, 1> copy() const {
        Tensor<T, 1> a(widths);
        a.strides = strides;
        a.offset = offset;
        a.data = make_shared<std::vector<T>>(*data);
        return a;
    }

    // constructor that takes widths as initializer list
    Tensor<T,1>(const std::initializer_list<size_t> a){
        assert( *(a.begin()) > 0 );

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    // costruttore which takes widths as a vector
    Tensor<T,1>(const std::vector<size_t> a){
        assert(a.size()==1);
        assert( *(a.begin()) > 0 );

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    // copy constructor
    Tensor<T,1>(const Tensor<T>& a) : widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){}

    //move constructor, tensor is passed as parameter and it is emptied of its data
    Tensor<T, 1>(Tensor<T>&& a) : widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){
        //we cannot change "a" metadata because they are immutable
        a.data = std::shared_ptr<std::vector<T>>();
    }

    // constructor that takes widths as initializer list and the relative vector of data
    Tensor<T, 1>(const std::initializer_list<size_t>& a, std::vector<T>& new_data) {
        assert(a.size() == 1 && new_data.size() == mult<size_t>(a));
        assert( *(a.begin()) > 0 );

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<new_data>;
        offset = 0;
    }

    // initialize with an array that will be represented as a ttensor
    void initialize(const std::initializer_list<T>& a){
        assert ( (a.size() == 1));
        data = make_shared<std::vector<T>>(a);
    }

    // get/set for data, giving back the refence you can also assign
    T& operator()(const initializer_list<size_t>& indices_l){
        std::vector<size_t> indices = indices_l;
        assert(indices.size() == 1);

        assert(indices[0] < widths[0] && indices[0] >= 0);
        assert(data);

        std::vector<size_t> indices_v = indices;
        size_t tmp = (indices_v[0] * strides[0]) + offset;
        return (*data)[tmp];
    }

    // get/set for data, giving back the refence you can also assign
    T& operator()(vector<int> indices){
        assert(indices.size() == 1);
        assert(indices[0] < widths[0] && indices[0] >= 0);
        assert(data);

        size_t tmp = (indices[0] * strides[0]) + offset;
        return (*data)[tmp];
    }

    //allow to set an element
    void set(const initializer_list<size_t> indices_l, const T& value){
        vector<size_t> indices = indices_l;
        assert(indices.size() == 1);
        assert(indices[0] < widths[0] && indices[0] >= 0);
        assert(data);

        size_t tmp = (indices[0] * strides[0]) + offset;
        (*data)[tmp] = value;
    }

    //allow to get an element
    T get(const initializer_list<size_t> indices_l){
        vector<size_t> indices = indices_l;
        assert(indices.size() == 1);
        assert(indices[0] < widths[0] && indices[0] >= 0);
        assert(data);

        size_t tmp = (indices[0] * strides[0]) + offset;
        return (*data)[tmp];
    }

    // window, takes index of dimension to be shrinked and the ends that will be accepted after operation.
    Tensor<T,1> window(const size_t& index, const size_t& start, const size_t& stop){
        assert(stop > start);
        assert(widths[index] > stop);
        assert(start >= 0);
        assert(data);

        Tensor<T, 1> a = Tensor<T, 1>(widths);

        a.widths[index] = stop - start + 1;
        a.offset += a.strides[index] * start;

        return a;
    }

private:

    // metadata : immutable
    std::vector<size_t> widths;
    std::vector<size_t> strides;
    size_t offset=0;

    //data : mutable
    std::shared_ptr<std::vector<T>> data;
};

//##########################################################################
//              TENSOR ITERATOR
//##########################################################################

template<class T, size_t rank>
class TensorIterator {
public:

    // takes one tensor and builds an iterator that starts from 0
    TensorIterator<T, rank>(Tensor<T, rank>& tensor) : ttensor(tensor) {
        indexes = std::vector<int>(ttensor.widths.size(), 0);
    };

    //copy constructor
    TensorIterator<T, rank>(const TensorIterator<T, rank>& old_iterator) : ttensor(old_iterator.ttensor), indexes(old_iterator.indexes) {};

    //constructor that takes indexes from which the iterator will starts, no check about correctness
    TensorIterator<T, rank>(Tensor<T, rank>& tensor, const std::vector<int>& starting_indexes) : indexes(starting_indexes), ttensor(tensor){};

    // read and write to an element pointed by iterator
    T& operator*() const {
        return ttensor(indexes);
    }

    // returns pointer to element pointed by iterator
    T* operator->() const {
        return &(ttensor(indexes));
    }

    // postfix operator
    TensorIterator<T, rank> operator++(int) {
        TensorIterator<T, rank> new_iterator = TensorIterator<T, rank>(*this);
        increment(1);
        return new_iterator;
    }

    // prefix operator
    TensorIterator<T, rank>& operator++() {
        increment(1);
        return *this;
    }

    TensorIterator<T, rank> operator--(int) {
        TensorIterator<T, rank> new_iterator = TensorIterator<T, rank>(*this);
        increment(-1);
        return new_iterator;
    }

    TensorIterator<T, rank>& operator--() {
        increment(-1);
        return *this;
    }

    bool operator==(const TensorIterator<T, rank>& other_iterator) const {
        return (&other_iterator.ttensor == ttensor && &other_iterator.indexes == indexes);
    }

    bool operator!=(const TensorIterator<T, rank>& other_iterator) const {
        return (&other_iterator.ttensor == &ttensor && other_iterator.indexes != indexes);
    }

    TensorIterator<T, rank>& operator+=(int inc) {
        increment(inc);
        return *this;
    }

    TensorIterator<T, rank>& operator-=(int dec) {
        increment(-dec);
        return *this;
    }

    TensorIterator<T, rank> operator+(int inc) const {
        TensorIterator<T, rank> new_iterator = TensorIterator<T, rank>(*this);
        new_iterator.increment(inc);
        return new_iterator;
    }

    TensorIterator<T, rank> operator-(int dec) const {
        TensorIterator<T, rank> new_iterator = TensorIterator<T, rank>(*this);
        new_iterator.increment(-dec);
        return new_iterator;
    }

    T& operator[](int access_index) const {
        std::vector<int> tmp_indexes = indexes;
        indexes = std::vector<int>(indexes.size(), 0);
        increment(access_index);
        T& tmp_ret = ttensor(indexes);
        indexes = tmp_indexes;
        return tmp_ret;
    }

    bool operator<(const TensorIterator<T, rank>& other_iterator) const {
        return (&other_iterator.ttensor == &ttensor && indexes < other_iterator.indexes);
    }

    bool operator>(const TensorIterator<T, rank>& other_iterator) const {
        return (&other_iterator.ttensor == &ttensor && indexes > other_iterator.indexes);
    }

    bool operator<=(const TensorIterator<T, rank>& other_iterator) const {
        return (&other_iterator.ttensor == &ttensor && indexes <= other_iterator.indexes);
    }

    bool operator>=(const TensorIterator<T, rank>& other_iterator) const {
        return (&other_iterator.ttensor == &ttensor && indexes >= other_iterator.indexes);
    }

    int operator-(const TensorIterator<T, rank>& other_iterator) const {
        assert(&other_iterator.ttensor == &ttensor);
        return (single_index() - other_iterator.single_index());
    }

private:
    Tensor<T, rank>& ttensor;
    std::vector<int> indexes;
    size_t sliding_index={};
    // we will to use int because we use negative numbers when we decrement

    int single_index() const {
        size_t single_index = 0;
        size_t acc_mult = 1;
        for (size_t i = indexes.size()-1; i >= 0; i--){
            single_index += indexes[i] * acc_mult;
            acc_mult *= ttensor.widths[i];
        }
        return single_index;
    }

    void increment(int index_inc) {
        size_t i = indexes.size() - 1;

        indexes[i] += index_inc;
        while (i > 0 && ((indexes[i] < 0) || (indexes[i] >= ttensor.widths[i]))) {
            if (indexes[i] < 0) {
                index_inc = ceil(indexes[i] / ttensor.widths[i]); 
            } else {
                index_inc = floor(indexes[i] / ttensor.widths[i]); 
            }
            indexes[i] = indexes[i] % ttensor.widths[i];
            indexes[i - 1] += index_inc;
            --i;
        }
    }
};

//##########################################################################
//              TENSOR ITERATOR FIXED
//##########################################################################
template<class T,size_t rank>
class TensorIteratorFixed{
public:

    // constructor that takes tensor, indexes from which it can start and index of dimension to slide
    TensorIteratorFixed<T, rank>(Tensor<T>& tensor, const std::vector<int>& starting_indexes, const size_t& sliding_index) :ttensor(tensor) {
        size_t indexes_size = starting_indexes.size();
        assert(indexes_size == tensor.widths.size());

        assert((sliding_index >= 0) && (sliding_index < indexes_size));


        indexes = std::vector<int>(starting_indexes);
        this->sliding_index = sliding_index;
    }

    // copy constructor
    TensorIteratorFixed<T, rank>(const TensorIteratorFixed<T, rank>& old_iterator) : ttensor(old_iterator.ttensor), indexes(old_iterator.indexes), sliding_index(old_iterator.sliding_index) {}

    // access read/write to element pointed by iterator
    T& operator*() const {
        return ttensor(indexes);
    }

    // turns back pointer to element pointed by iterator
    T* operator->() const {
        return &(this->ttensor(this->indexes));
    }

    TensorIteratorFixed<T, rank> operator++(int) {
        TensorIteratorFixed<T, rank> new_iterator = TensorIteratorFixed<T, rank>(*this);
        increment(1);
        return new_iterator;
    }

    TensorIteratorFixed<T, rank>& operator++() {
        increment(1);
        return (*this);
    }

    TensorIteratorFixed<T, rank> operator--(int) {
        TensorIteratorFixed<T, rank> new_iterator = TensorIteratorFixed<T, rank>(*this);
        increment(-1);
        return new_iterator;
    }

    TensorIteratorFixed<T, rank>& operator--() {
        increment(-1);
        return this;
    }

    bool operator==(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) && other_iterator.indexes == indexes && other_iterator.sliding_index == sliding_index);
    }

    bool operator!=(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) &&
        ( other_iterator.indexes != indexes || other_iterator.sliding_index != sliding_index) );
    }

    TensorIteratorFixed<T, rank>& operator+=(int inc) {
        increment(inc);
        return *this;
    }

    TensorIteratorFixed<T, rank>& operator-=(int dec) {
        increment(-dec);
        return *this;
    }

    TensorIteratorFixed<T, rank> operator+(int inc) const {
        TensorIteratorFixed<T, rank> new_iterator = TensorIteratorFixed<T, rank>(*this);
        new_iterator.increment(inc);
        return new_iterator;
    }

    TensorIteratorFixed<T, rank> operator-(int dec) const {
        TensorIteratorFixed<T, rank> new_iterator = TensorIteratorFixed<T, rank>(*this);
        new_iterator.increment(-dec);
        return new_iterator;
    }

    T& operator[](int access_index) const {
        std::vector<int> tmp_indexes = indexes;
        tmp_indexes[sliding_index] = access_index;
        return ttensor(tmp_indexes);
    }

    bool operator<(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) &&
                (sliding_index == other_iterator.sliding_index) &&
                check_indexes_equality(other_iterator, sliding_index ) &&
                (indexes[sliding_index] < other_iterator.indexes[sliding_index]) );
    }

    bool operator>(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) &&
                (sliding_index == other_iterator.sliding_index) &&
                check_indexes_equality(other_iterator) &&
                (indexes[sliding_index] > other_iterator.indexes[sliding_index]) );
    }

    bool operator<=(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) &&
                (sliding_index == other_iterator.sliding_index) &&
                check_indexes_equality(other_iterator) &&
                (indexes[sliding_index] <= other_iterator.indexes[sliding_index]) );
    }

    bool operator>=(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) &&
                (sliding_index == other_iterator.sliding_index) &&
                check_indexes_equality(other_iterator) &&
                (indexes[sliding_index] >= other_iterator.indexes[sliding_index]) );
    }

    int operator-(const TensorIteratorFixed<T, rank>& other_iterator) const {
        assert( (&other_iterator.ttensor == &ttensor) &&
                (other_iterator.sliding_index == sliding_index) &&
                check_indexes_equality(other_iterator) );
        return (indexes[sliding_index] - other_iterator.indexes[sliding_index]);
    }

private:

    Tensor<T, rank>& ttensor;
    std::vector<int> indexes;
    size_t sliding_index;

    void increment(const int& index_inc) {
        indexes[sliding_index] += index_inc;
    }

    bool check_indexes_equality(const TensorIteratorFixed<T, rank> other_iter, const size_t index_ignore) const {
        int i = 0;
        while (i < indexes.size() && (i == index_ignore || indexes[i] == other_iter.indexes[i]) ) {
            i++;
        }
       return (i == indexes.size());
    }
};


#endif //TENSORLIB_TENSORLIB_H