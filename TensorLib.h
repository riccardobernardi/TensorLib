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
    //costruttore con le widths
    Tensor<T, rank>(std::initializer_list<size_t>&& a){
        // cout << "costruttore : Tensor<T>(std::initializer_list<size_t>&& a)" << endl;
        //TODO decidere se rank può essere 0
        if (rank!=0){
            assert(a.size()==rank);
        }

        widths = a;
        strides = cummult<size_t>(widths,1);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
        offset = 0;
    }

    // copy constructor
    Tensor<T, rank>(const Tensor<T, rank>& a){
        widths = a.widths;
        strides = a.strides;
        data = a.data;
        offset = a.offset;
    }

    //move constructor, se forniamo il move constructor dobbiamo permettere di reinserire il vettore di dati, infatti il tensore che viene passato come parametro al move constructor rimarrà vuoto.
    Tensor<T, rank>(const Tensor<T, rank>&& a){
        widths = a.widths;
        strides = a.strides;
        data = a.data;
        offset = a.offset;

        a.widths = std::vector<size_t>();
        a.strides = std::vector<size_t>();
        a.data = std::shared_ptr<std::vector<T>>();             //non è necessario fare altre operazioni sul vecchio shared_pointer (per fare in modo che decrementi il contatore di pointers attivi)
        a.offset = 0;                                           //poichè l'operatore = è overloadato e ci pensano loro
    }



    //TODO lo facciamo il move constructor?

    //costruttore con le widths e i dati
    Tensor<T, rank>(std::initializer_list<size_t>&& new_widths, std::vector<T>& new_data){
       //TODO decidere se rank può essere 0
        if (rank!=0){
            assert(new_widths.size()==rank);
        }
        size_t full_size = std::accumulate(new_widths.begin(), new_widths.end(), 1, std::multiplies<size_t>());
        asser(full_size == new_data.size())
        widths = new_widths;
        strides = cummult<size_t>(widths,1);
        offset = 0;
        initialize(new_data); //TODO è giusto passare il parametro così?
    }

    //TODO cntrollo su width piuttosto che su su data.size() ??
    //initialize with an array that will be represented as a tensor
    void initialize(std::initializer_list<T>&& a){
        if ( (!data) || (a.size() == (*data).size()) ){
            data = make_shared<std::vector<T>>(a);
        }else{
            // TODO gestire con gli assert o altri tipi di errore?
            cout << "Una volta inizializzato il vettore non può essere modificato nelle dimensioni!" << endl;
        }
    }

    //ritornando la reference si lascia la possibilità di settare il valore dell'elemento ritornato
    T& operator()(initializer_list<size_t> indices){
        assert(indices.size() == widths.size());

        std::vector<size_t> indices_v = indices;
        size_t tmp = 0;
        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            tmp += indices_v[i] * strides[i];
        }

        tmp += offset;

        return (*data)[tmp];
    }

    //questo lo teniamo però anche l'operatore parentesi può essere usato per settare
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
        
        Tensor<T, rank-1> a = Tensor<T, rank-1>();
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
        a.strides = cummult(new_width);
        a.data = data;

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

        return a;
    }
    
private:

    // metadata : immutable
    std::vector<size_t> widths;
    std::vector<size_t> strides;
    size_t offset=0;

    //data : mutable
    std::shared_ptr<std::vector<T>> data;

    //default constructor 
    Tensor<T, rank>() : width(), strides(), data() {      //lo usiamo internamente per comodità (nelle slice/window), non ha senso di essere pubblico
        offset = 0;
    }
};





#endif //TENSORLIB_TENSORLIB_H
