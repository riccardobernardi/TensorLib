//
// Created by rr on 29/10/2019.
//

#ifndef TENSORLIB_TENSORLIB_H
#define TENSORLIB_TENSORLIB_H

#include <vector>
#include <algorithm>
#include <type_traits>
#include "utilities.h"
#include<cstdarg>


using namespace std;

template<class T = int, size_t rank=0>
class Tensor {
public:
    // when the rank is not specified
    Tensor<T>(std::initializer_list<size_t>&& a){
        // cout << "costruttore : Tensor<T>(std::initializer_list<size_t>&& a)" << endl;
        if (rank!=0){
            assert(a.size()==rank);
        }
        _width = a;
        _strides = cummult<size_t>(_width,1);
        _rank = a.size();
        _size = cummult(_width)[0];
    }

    // when the rank is specified
    Tensor<T>(const std::vector<size_t>& a){
        // cout << "costruttore : Tensor<T>(std::vector<size_t>& a)" << endl;
        if (rank!=0){
            assert(a.size()==rank);
        }
        _width = a;
        _strides = cummult<size_t>(_width,1);
        _rank = a.size();
        _size = cummult(_width)[0];
    }

    // move constructor
    Tensor<T>(Tensor<T>&& a){
        // cout << "costruttore : Tensor<T>(Tensor<T>&& a)" << endl;
        if (rank!=0){
            assert(a._rank==rank);
        }
        _width = a._width;
        _strides = a._strides;
        _rank = a._rank;
        _size = a._size;
        _vec = a._vec;
    }

    Tensor<T>(Tensor<T>& a){
        // cout << "costruttore : Tensor<T>(Tensor<T>& a)" << endl;
        if (rank!=0){
            assert(a._rank==rank);
        }
        _width = a._width;
        _strides = a._strides;
        _rank = a._rank;
        _size = a._size;
        _vec = a._vec;
    }

    // initialize with an array that will be represented as a tensor
    void initialize(std::initializer_list<T>&& a){
        // assert(a.size() == _size);
        // questo controllo serve perchè il vettore dev'essere immutable una volta creato per la prima volta
        // assert( (_vec.get() == nullptr) || (a.size() == _vec.get()->size()) );
        if ( (_vec.get() == nullptr) || (a.size() == _vec.get()->size()) ){
            _vec = make_shared<std::vector<T>>(a);
        }else{
            cout << "Una volta inizializzato il vettore non può essere modificato nelle dimensioni!" << endl;
        }
        // cout << "operazione pericolosa" << endl;
        // cout << "controlliamo che il vettore sia correttamente istanziato" << _vec.get()->at(0) << endl;
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
        cout << "vvvvvvvvvvvvvvvvvvvvvv" << endl;
        cout << "private " << "_rank " << _rank << endl;
        cout << "private " << "_size " << _size << endl;
        cout << "private size del vettore " << _vec.get()->size() << endl;
        cout << "private " << "size del _default " << _default.size() << endl;
        cout << "private " << "size del _old_dimensions " << _old_dimensions.size() << endl;
        cout << "private " << "flattened dimensions " << _flattened_dim << endl;
        for(auto i : _default){
            cout << "_default is " << get<0>(i) << " " << get<1>(i) << endl;
        }
        cout << "^^^^^^^^^^^^^^^^^^^^^^" << endl;
    }

    /*Tensor<T> operator=(Tensor<T>&& a){
        return Tensor<T>(a);
    }*/

    T operator()(initializer_list<size_t> a){
        std::vector<size_t> b=a;
        for(auto i: _default){
            b.insert(b.begin() + get<0>(i), get<1>(i));
        }

        if(_old_dimensions.size() == 0){
            assert(_strides.size() == b.size());
        }else{
            // In this case the tensor was flattened so i need to replace the flattened parts with the real ones
            std::vector<size_t> query;
            for(int i = 0; i<b.size(); ++i ){
                if(i != _flattened_dim){
                    query.push_back(b[i]);
                }else{
                    // this is the case in which we found the dimension to be substituted
                    assert(_old_dimensions.size() - b.size() > 0);
                    for(int j = 0; j <= _old_dimensions.size() - b.size(); ++j){
                        query.push_back(b[i] % _old_dimensions[i+j]);
                        // cout << "aggiungo: " << b[i] % _old_dimensions[i+j] << endl;
                    }
                }
            }

            /*cout << "dimension of strides: " << _strides.size() << endl;
            cout << "dimension of query: " << query.size() << endl;
            for (auto i : query){
                cout << "my query value in query is : " << i << endl;
            }

            for (auto i : _strides){
                cout << "my strides value in strides is : " << i << endl;
            }*/

            b = query;
        }
        return (_vec.get()->at(sum(mult(_strides, b))));
    }

    Tensor<T> slice(size_t index, size_t value){
        // have to check if there's another one index equal to mine, in this case i've to increase mine
        // why do i have to increase and not to decrease? Because im working in a subset and for example if i take out the second and again the second then the third(the second second) is the third!
        if (_default.size() != 0){
            cout << "Ho notato che hai fatto slicing a catena su un vettore, questa feature non è ancora testata!!";
            for(auto j : _default){
                for(auto i: _default){
                    if(index == get<0>(i)){
                        index++;
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

    Tensor<T> flatten(size_t start, size_t stop){
        if(stop - start == _rank - 1){
            cout << "ATTENZIONE: stai tentando di passare da un tensore ad un array completamente lineare, devo ancora finire di implementare questa feature:)";
        }
        std::vector<size_t> new_width;
        size_t tmp=1;
        size_t flat=-1;

        /*for(auto i: _width){
            cout << "vecchia dimensione" << i << endl;
        }*/

        for(int i=0;i<_width.size();++i){
            if (i<start || i>stop){
                new_width.push_back(_width[i]);
            }else{
                tmp*=_width[i];
                if(i==stop){
                    new_width.push_back(tmp);
                    flat = new_width.size() - 1;
                }
            }
        }

        /*for(auto i: new_width){
            cout << "nuova dimensione" << i << endl;
        }*/

        Tensor<T> result = Tensor<T>(new_width);
        result._flattened_dim = flat;
        // cout << "dopo aver fatto il flat lo segno nel nuovo tensore come: " << result._flattened_dim << endl;
        result._old_dimensions = _old_dimensions.size() == 0?_width:_old_dimensions;
        result._vec = _vec;
        // TO_DO: devo mettere lo stesso controllo che c'è anche su old_dimensions perchè se faccio doppio flatten si sminchia
        result._strides = _strides;
        //cout << "assegnazione del vettore delle vechchie dims " << result._old_dimensions.size() << endl;
        // result.print_privates();
        return result;
    }

    inline typename std::vector<T>::iterator begin() noexcept {
        return _vec.get()->begin();
    }

    inline typename std::vector<T>::iterator end() noexcept {
        return _vec.get()->end();
    }

    inline typename std::vector<T>::const_iterator cbegin() noexcept {
        return _vec.get()->cbegin();
    }

    inline typename std::vector<T>::const_iterator cend() noexcept {
        return _vec.get()->cend();
    }

private:

    size_t _rank=0;
    size_t _size=0;
    size_t _flattened_dim=-1000;

    // when you do slicing then there is an index that it is defaulted
    std::vector<tuple<size_t, size_t>> _default;
    std::vector<size_t> _old_dimensions;

    std::shared_ptr<std::vector<T>> _vec;
    std::vector<size_t> _strides;
    std::vector<size_t> _width;
};





#endif //TENSORLIB_TENSORLIB_H
