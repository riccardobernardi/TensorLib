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
        //TODO decidere se rank può essere 0
        if (rank!=0){
            assert(a.size()==rank);
        }

        widths = a;
        strides = cummult<size_t>(widths,1);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    // when the rank is specified
    Tensor<T>(const std::vector<size_t>& a){
        // cout << "costruttore : Tensor<T>(std::vector<size_t>& a)" << endl;
        if (rank!=0){
            assert(a.size()==rank);
        }
        widths = a;
        strides = cummult<size_t>(widths,1);
    }

    // copy constructor
    Tensor<T>(const Tensor<T>& a){
        widths = a.widths;
        strides = a.strides;
        data = a.data;
        offset = a.offset;
    }

    // initialize with an array that will be represented as a tensor
    void initialize(std::initializer_list<T>&& a){
        if ( (data.get() == nullptr) || (a.size() == data.get()->size()) ){
            data = make_shared<std::vector<T>>(a);
        }else{
            // cout << a.size() << data.get()->size() << endl;
            cout << "Una volta inizializzato il vettore non può essere modificato nelle dimensioni!" << endl;
        }
    }

    T operator()(initializer_list<size_t> indices){
        assert(indices.size() == widths.size());

        std::vector<size_t> indices_v = indices;
        size_t tmp = 0;

        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            cout << "stride: " << strides[i] << endl;
            tmp += indices_v[i] * strides[i];
            // cout << "value: " << tmp << endl;
        }
        tmp += offset;
        // cout << "+tmp: " << tmp << endl;

        // cout << tmp << endl;
        assert(data.get() != nullptr);
        assert(data.get()->size() == 36);

        /*cout << "result:" << data.get()->size() << endl;
        cout << "value:" << tmp << endl;
        cout << "result:::::" << data.get()->at(tmp) << endl;*/

        return (data.get()->at(tmp));
        //return (*data)[tmp];      //versione alternativa
    }

    void set(initializer_list<size_t> indices, T& value){
        assert(indices.size() == widths.size());

        std::vector<size_t> indices_v = indices;
        size_t tmp = 0;
        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            tmp += indices_v[i] * strides[i];
        }

        tmp += offset;

        (*data)[tmp] = value;
    }

    Tensor<T> slice(const size_t&  index, const size_t& value){
        assert(index >= 0 && index < widths.size());
        assert(value >= 0 && value < widths[index]);
        
        if(rank == 1){
            //TODO caso particolare in cui si crea un tensore di rango 0
        }
        
        Tensor<T> a = Tensor<T>(widths);
        /*
        tmp_widths = erase<size_t>(widths, index);
        Tensor<T, rank-1> a = Tensor<T, rank-1>(tmp_widths);         //versione inefficiente ma corretta, crea un vettore vuoto 
        a.strides = erase<size_t>(strides, index);                   // il problema di questa è che se nel costruttore controllo che (*data).size() sia conforme con width, 
        a.offset += (strides[index] * value);                        //mi becco un errore perchè facendo lo slice ho degli elementi nel vettore che non c'entrano, quindi (*data).size()>mult(width)
        a.data = data;                                               // con l'altra versione ho dubbi: non sfruttiamo il fatto che sappiamo il rank a tempo di compilazione
        */
        a.widths = erase<size_t>(widths, index);
        a.strides = erase<size_t>(strides, index);
        a.offset += (strides[index] * value);
        a.data = data;
        assert(data.get() != nullptr);
        return a;
    }

    Tensor<T> flatten(const size_t& start, const size_t& stop){  //estremi inclusi
        assert(start >= 0 && start < widths.size());
        assert(stop >= 0 && stop < widths.size());

        std::vector<size_t> new_width;
        size_t tmp=1;

        if ( (rank - (stop - start +1)) == 0 ){
            //TODO caso in cui il risultante è un tensore di grado 0
        }

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

        Tensor<T> a = Tensor<T>(new_width); //qui non conosciamo il rank a tempo di compilazione perchè dipende da start e width

        // TODO opt
        a.strides = cummult<size_t>(new_width,1);
        a.data = data;
        a.offset = offset;

        assert(data.get() != nullptr);
        return a;
    }

    Tensor<T> window(const size_t& index, const size_t& start, const size_t& stop){
        assert(stop > start);
        assert(widths[index] > stop);
        assert(start >= 0);

        //TODO gestire caso in cui genero un tensore di rank 0

        Tensor<T> a = Tensor<T>(widths);

        a.widths[index] = stop - start + 1;
        a.offset += a.strides[index] * start;

        assert(data.get() != nullptr);
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





#endif //TENSORLIB_TENSORLIB_H
