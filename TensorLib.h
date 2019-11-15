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
//              dichiarazioni
//##########################################################################

template<class T, size_t rank>
class TensorIterator;

template<class T, size_t rank>
class TensorIteratorFixed;

template<class T, size_t rank>
TensorIterator<T, rank> operator+(size_t n, TensorIterator<T, rank> iter){
    return iter + n;
};

template<class T, size_t rank>
TensorIteratorFixed<T, rank> operator+(size_t n, TensorIteratorFixed<T, rank> iter){
    return iter + n;
};

//##########################################################################
//              TENSORE FISSO
//##########################################################################

// TODO controlli sulla larghezza del vettore
template<class T = size_t, size_t rank=0>
class Tensor {
public:

    friend class TensorIterator<T, rank>;

    //TODO serve?
    friend class Tensor<T, rank + 1>;

    /*TensorIterator<T, 0> it(){
        return TensorIterator<T, 0>(*this);
    }*/

    TensorIterator<T, rank> begin(){
        std::vector<size_t> ind(widths.size(), 0);
        return TensorIterator<T,rank>(*this,ind);
    }

    TensorIterator<T, rank> end(){
        std::vector<int> ind(widths.size(), 0);
        ind.at(0) = widths[0];
        return TensorIterator<T,rank>(*this,ind);
    }

    Tensor<T, rank> copy(){
        Tensor<T, rank> a(widths);
        a.strides = strides;
        a.offset = offset;
        a.data = make_shared<std::vector<T>>(*data);
        return a;
    }

    Tensor<T,rank>(std::initializer_list<size_t> a){
        // cout << "costruttore : Tensor<T>(std::initializer_list<size_t>&& a)" << endl;
        assert(a.size()==rank);

        cout << "sto usando il generico con valore rank:" << rank << endl;

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    //TODO secondo cek bisogna toglierla
    Tensor<T,rank>(std::initializer_list<size_t>& a){
        // cout << "costruttore : Tensor<T>(std::initializer_list<size_t>&& a)" << endl;
        assert(a.size()==rank);

        cout << "sto usando il generico con valore rank:" << rank << endl;

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    // copy constructor
    Tensor<T, rank>(const Tensor<T, rank>& a) : widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){}

    //move constructor, se forniamo il move constructor dobbiamo permettere di reinserire il vettore di dati, infatti il tensore che viene passato come parametro al move constructor rimarrà vuoto.
    Tensor<T, rank>(Tensor<T, rank>&& a) : widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){
        /* non possiamo cambiare i metadati dell'altro tensore perchè i metadati sono immutable
        a.widths = std::vector<size_t>();
        a.strides = std::vector<size_t>();
        a.offset = 0;
                                                                 //non è necessario fare altre operazioni sul vecchio shared_pointer (per fare in modo che decrementi il contatore di pointers attivi)
        */                                                       //poichè l'operatore = è overloadato e ci pensano loro
        a.data = std::shared_ptr<std::vector<T>>();
    }

    //costruttore che prende width e data

    Tensor<T>(std::initializer_list<size_t>& a, std::vector<T> new_data) : {
        assert(a.size() > 0 && new_data.size() == mult(a));

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<new_data>;
        offset = 0;
    }

    // initialize with an array that will be represented as a ttensor
    void initialize(std::initializer_list<T>& a){
        assert ( !data || (a.size() == mult(widths)));
        data = make_shared<std::vector<T>>(a);
    }

    //ritornando la reference si lascia la possibilità di settare il valore dell'elemento ritornato
    T& operator()(initializer_list<size_t> indices){
        assert(indices.size() == widths.size());
        //TODO serve?
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

    Tensor<T, rank - 1> slice(const size_t&  index, const size_t& value){
        assert(index >= 0 && index < widths.size());
        assert(value >= 0 && value < widths[index]);
        //TODO serve?
        assert(data);

        Tensor<T, rank - 1> a = Tensor<T, rank - 1>();
        a.widths = erase<size_t>(widths, index);
        a.strides = erase<size_t>(strides, index);
        a.offset += (strides[index] * value);
        a.data = data;


        return a;
    }

    //flatten singola, prende l'indice della dimensione di sinistra, es: dim = [2,3,5,1], flatten(2) prende le dimensioni larghe 5 e 1
    Tensor<T, rank - 1> flatten(const size_t& start){  //flatten tra start e start+1
        assert(start >= 0 && start < widths.size()-1);
        //TODO serve?
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

    Tensor<T> multiFlatten(const size_t& start, const size_t& stop){  //estremi inclusi
        assert(start >= 0 && start < widths.size());
        assert(stop >= 0 && stop < widths.size());
        assert((rank - (stop - start +1)) != 0) //non si può tornare un tensore di rank 0
        //TODO serve?
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

        Tensor<T> a = Tensor<T>(new_width); //qui non conosciamo il rank a tempo di compilazione perchè dipende da start e width

        a.strides = cummult(new_width);
        a.data = data;
        a.offset = offset;

        return a;
    }

    Tensor<T, rank> window(const size_t& index, const size_t& start, const size_t& stop){
        assert(stop > start);
        assert(widths[index] > stop);
        assert(start >= 0);


        //TODO serve?
        assert(data);

        Tensor<T, rank> a = Tensor<T, rank>(widths);

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
    Tensor<T, rank>() : widths(), strides(), data(), offset() {}      //lo usiamo internamente per comodità (nelle slice/window), non ha senso di essere pubblico

};

//##########################################################################
//              TENSORE DINAMICO
//##########################################################################

// tutti i tensori con dimensione non specificata staticamente vanno qui
// ha gli stessi metodi di quello sopra
///////////////////////TENSORE DINAMICO
// inserire assert per controllare operazioni illecite
template<class T>
class Tensor<T,0> {
public:
    friend class TensorIterator<T, 0>;

    //TODO fare Tensor<T, r> friend

    /*TensorIterator<T, 0> it(){
        return TensorIterator<T, 0>(*this);
    }*/

    TensorIterator<T, 0> begin(){
        std::vector<int> ind(widths.size(), 0);
        return TensorIterator<T,0>(*this,ind);
    }

    TensorIterator<T, 0> end(){
        std::vector<int> ind(widths.size(), 0);
        ind.at(0) = widths[0];
        return TensorIterator<T,0>(*this,ind);
    }

    Tensor<T, 0> copy(){
        Tensor<T, 0> a(widths);
        a.strides = strides;
        a.offset = offset;
        a.data = make_shared<std::vector<T>>(*data);
        return a;
    };

    Tensor<T>(std::initializer_list<size_t> a){
        // cout << "costruttore : Tensor<T>(std::initializer_list<size_t> a)" << endl;
        assert(a.size() > 0);
        // cout << "stiamop usando la spec" << endl;


        widths = a;
        strides = cummult(widths);
        // cout <<"www"<< widths[0] << endl;
        // cout <<"sss"<< strides[0] <<endl;
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    //TODO secondo cek bisogna toglierla
    Tensor<T>(std::initializer_list<size_t>& a){
        // cout << "costruttore : Tensor<T>(std::initializer_list<size_t>& a)" << endl;
        assert(a.size() > 0);

        // cout << "stiamop usando la spec" << endl;


        widths = a;
        strides = cummult(widths);
        // cout <<"www"<< widths[0] << endl;
        // cout <<"sss"<< strides[0] <<endl;
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
        offset = 0;
    }

    // copy constructor
    Tensor<T>(const Tensor<T>& a): widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){}

    //move constructor, se forniamo il move constructor dobbiamo permettere di reinserire il vettore di dati, infatti il tensore che viene passato come parametro al move constructor rimarrà vuoto.
    Tensor<T, rank>(Tensor<T, rank>&& a) : widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){
        /* non possiamo cambiare i metadati dell'altro tensore perchè i metadati sono immutable
        a.widths = std::vector<size_t>();
        a.strides = std::vector<size_t>();
        a.offset = 0;
                                                                 //non è necessario fare altre operazioni sul vecchio shared_pointer (per fare in modo che decrementi il contatore di pointers attivi)
        */                                                       //poichè l'operatore = è overloadato e ci pensano loro
        a.data = std::shared_ptr<std::vector<T>>();
    }

    //costruttore che prende width e data

    Tensor<T>(std::initializer_list<size_t>& a, std::vector<T> new_data) : {
        assert(a.size() > 0 && new_data.size() == mult(a));

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<new_data>;
        offset = 0;
    }


/*    // la costruzione è tipo Tensor<int> a()
    // voglio ottenere Tensor<int,3> b = a
    template<int rank>
    Tensor<T>(const Tensor<T, rank>& a){
        widths = a.widths;
        strides = a.strides;
        data = a.data;
        offset = a.offset;
    }*/

/*    template<class S, int rank>
    Tensor<S, rank> convert(){
        Tensor<S,rank> a(widths);
        a.strides = strides;
        a.data = data;
        a.offset = offset;
        return a;
    }*/

    // initialize with an array that will be represented as a ttensor
    void initialize(std::initializer_list<T>&& a){
        assert ( !data || (a.size() == mult(widths)));
        data = make_shared<std::vector<T>>(a);
    }

    T& operator()(initializer_list<size_t> indices){
        assert(indices.size() == widths.size());

        std::vector<size_t> indices_v = indices;
        size_t tmp = 0;

        for(size_t i=0; i< indices_v.size(); ++i){
            assert(indices_v[i] < widths[i] && indices_v[i] >= 0);
            // cout << "stride: " << strides[i] << endl;
            tmp += indices_v[i] * strides[i];
            // cout << "value: " << tmp << endl;
        }
        tmp += offset;
        // cout << "+tmp: " << tmp << endl;

        // cout << tmp << endl;
        assert(data);

        //TODO WTF?
        assert(data.get()->size() == 36);

        /*cout << "result:" << data.get()->size() << endl;
        cout << "value:" << tmp << endl;
        cout << "result:::::" << data.get()->at(tmp) << endl;*/

        return (*data)[tmp];
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
        assert(widths.size() > 1);  // si può fare la slice solo di tensori con rank > 1

        //TODO serve?
        assert(data);

        Tensor<T> a = Tensor<T>(widths);
        /*
        tmp_widths = erase<size_t>(widths, index);
        Tensor<T, rank0> a = Tensor<T, rank0>(tmp_widths);         //versione inefficiente ma corretta, crea un vettore vuoto
        a.strides = erase<size_t>(strides, index);                   // il problema di questa è che se nel costruttore controllo che (*data).size() sia conforme con width,
        a.offset += (strides[index] * value);                        //mi becco un errore perchè facendo lo slice ho degli elementi nel vettore che non c'entrano, quindi (*data).size()>mult(width)
        a.data = data;                                               // con l'altra versione ho dubbi: non sfruttiamo il fatto che sappiamo il rank a tempo di compilazione
        */
        a.widths = erase<size_t>(widths, index);
        a.strides = erase<size_t>(strides, index);
        a.offset += (strides[index] * value);
        a.data = data;

        return a;
    }


    //flatten singola, prende l'indice della dimensione di sinistra, es: dim = [2,3,5,1], flatten(2) prende le dimensioni larghe 5 e 1
    Tensor<T> flatten(const size_t& start){  //flatten tra start e start+1
        assert(start >= 0 && start < widths.size() - 1);
        assert(widths.size() >= 2);
        //TODO serve?
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

        Tensor<T> a = Tensor<T>(new_width); //qui non conosciamo il rank a tempo di compilazione perchè dipende da start e width

        // TODO opt
        a.strides = cummult(new_width);
        a.data = data;
        a.offset = offset;


        return a;
    }

    Tensor<T> multiflatten(const size_t& start, const size_t& stop){  //estremi inclusi
        assert(start >= 0 && start < widths.size());
        assert(stop >= 0 && stop < widths.size());
        assert(widths.size() >= 2);     //forse non serve se facciamo l'assert sotto
        assert((widths.size() - (stop - start +1)) != 0) //non si può tornare un tensore di rank 0

        //TODO serve?
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

        Tensor<T> a = Tensor<T>(new_width); //qui non conosciamo il rank a tempo di compilazione perchè dipende da start e width

        // TODO opt
        a.strides = cummult(new_width);
        a.data = data;
        a.offset = offset;

        assert(data.get() != nullptr);
        return a;
    }

    Tensor<T> multiFlatten(const size_t& start, const size_t& stop){  //estremi inclusi
        assert(start >= 0 && start < widths.size());
        assert(stop >= 0 && stop < widths.size());
        // assert((stop-start) == 1);

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

        Tensor<T> a = Tensor<T>(new_width); //qui non conosciamo il rank a tempo di compilazione perchè dipende da start e width

        // TODO opt
        a.strides = cummult(new_width);
        a.data = data;
        a.offset = offset;


        return a;
    }

    Tensor<T> window(const size_t& index, const size_t& start, const size_t& stop){
        assert(stop > start);
        assert(widths[index] > stop);
        assert(start >= 0);
        //TODO serve?
        assert(data);

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
};






























//##########################################################################
//              ITERATORE T. RANK ARBITRARIO NON STATICO!!!!
//##########################################################################

//gli operatori di confronto danno sempre false se i tensori referenziati dagli iteratori non sono uguali
template<class T, size_t rank>
class TensorIterator {
public:

    TensorIterator<T, rank>(Tensor<T, rank>& tensor) : ttensor(tensor) {
        indexes = std::vector<int>(ttensor.widths.size(), 0);
    }

    TensorIterator<T, rank>(Tensor<T, rank>& tensor, std::vector<int>& starting_indexes) : indexes(starting_indexes), ttensor(tensor){}

/*    TensorIterator<T>(const TensorIterator<T>& old_iterator) {
        ttensor = old_iterator.ttensor;
        indexes = old_iterator.indexes;
    }*/

    T& operator*() const {
        return ttensor(indexes);
    }

    //TODO se faccio il pointer di una reference funziona?
    T* operator->() const {
        return &(ttensor(indexes));
    }

    TensorIterator<T, rank> operator++(int) {
        //crea nuovo, incrementa me e ritorna l'altro
        TensorIterator<T, rank> new_iterator = TensorIterator<T, rank>(*this);
        increment(1);
        return new_iterator;
    }

    TensorIterator<T, rank>& operator++() {
        //incrementa me e ritorna la referenza
        increment(1);
        return *this;
    }

    TensorIterator<T, rank> operator--(int) {
        //crea nuovo, decrementa me e ritorna l'altro
        TensorIterator<T, rank> new_iterator = TensorIterator<T, rank>(*this);
        increment(-1);
        return new_iterator;
    }

    TensorIterator<T, rank>& operator--() {
        //decrementa me e ritorna la referenza
        increment(-1);
        return *this;
    }

    //TODO check delle references dei tensori che abbiano stesso indirizzo
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
    //con size_t non si può lavorare con valori negativi, a noi serve int perchè per il decremento degli indici li mettiamo temporaneamente negativi

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
        //rimani dentro finchè gli indici sono fuori dal range, cioè finche l'incremento deve propagarsi all'indice superiore
        while ( i > 0 && ((indexes[i] < 0) || (indexes[i] >= ttensor.widths[i])) ) {
            if (indexes[i] < 0) {
                // TODO controllare che la divisione funzioni, ed: divisione intera o no?
                index_inc = ceil(indexes[i] / ttensor.widths[i]); // numero di volte in cui viene attraversato (in negativo) l'intervallo dato dalla width = numero da decrementare all'indice a sinistra
            } else {
                index_inc = floor(indexes[i] / ttensor.widths[i]);
            }
            indexes[i] = indexes[i] % ttensor.widths[i];
            indexes[i-1] += index_inc;
            --i;
        }
        //controllo overflow


        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        // assert( indexes[0] >= 0 && (indexes[0] < ttensor.widths[0]) );

        /* vechia versione solo positiva
        this->indexes[last_index] += index_inc;
        for (size_t i = last_index; i > 0; i--) {
            //controllare che faccia divisione intera
            this->indexes[i-1] += ( this->indexes[i] / this->ttensor.widths[i]);
            this->indexes[i] = ( this->indexes[i] % this->ttensor.widths[i]);
        }
        //controllo overflow
        assert(this->indexes[0] >= this->ttensor.widths[0]);
        */
    }
};

//##########################################################################
//              specializzazioni
//##########################################################################

// tensore di una dimensione, alias vettore, serve per la slice
// non ha la flatten
template<class T>
class Tensor<T,1> {
public:

    template<typename, typename> friend class Tensor;

    Tensor<T,1>(std::initializer_list<size_t> a){
        // cout << "costruttore : Tensor<T, rank>(std::initializer_list<size_t>&& a)" << endl;
        //TODO decidere se rank può essere 0

        // cout << "stiamop usando la spec" << endl;


        widths = a;
        strides = cummult(widths);
        // cout <<"www"<< widths[0] << endl;
        // cout <<"sss"<< strides[0] <<endl;
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    //TODO secondo ceck non serve
    Tensor<T,1>(std::initializer_list<size_t>& a){
        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    // copy constructor
    Tensor<T,1>(const Tensor<T>& a) : widths(a.widths), strides(a.strides), data(a.data), offset(a.offset){}

    // initialize with an array that will be represented as a ttensor
    void initialize(std::initializer_list<T>&& a){
        assert(!data || (a.size() == mult(widths) ) );
        data = make_shared<std::vector<T>>(a);
    }

    T& operator()(initializer_list<size_t> indices){
        assert(indices.size() == 1);

        assert(indices_v[0] < widths[0] && indices_v[0] >= 0);

        std::vector<size_t> indices_v = indices;
        size_t tmp = (indices_v[0] * strides[0]) + offset;
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

    Tensor<T,1> window(const size_t& index, const size_t& start, const size_t& stop){
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

//##########################################################################
//              FIXED ITERATOR
//##########################################################################

//tutti gli operatori di confronto ritornano false se il tensore riferito non è lo stesso o
//se la dimensione lungo la quale si scorre non è la stessa o
//se gli indici fissati non sono gli stessi
template<class T, size_t rank>
class TensorIteratorFixed{
public:

    TensorIteratorFixed<T, rank>(const Tensor<T>& tensor, const std::vector<int>& starting_indexes, const size_t& sliding_index) {
        size_t indexes_size = starting_indexes.size();
        assert(indexes_size == tensor.widths.size());

        assert((sliding_index >= 0) && (sliding_index < indexes_size));

        //TODO decidere se fare il controllo sugli indici all'interno
        /*
        for (size_t i = 0; i < indexes_size; i++){
            assert(starting_indexes[i] < tensor.widths[i] );
        }
        */

        ttensor = tensor;
        indexes = std::vector<int>(starting_indexes);
        sliding_index = sliding_index;
    }

    TensorIteratorFixed<T, rank>(const TensorIteratorFixed<T, rank>& old_iterator) : ttensr(old_iterator.ttensor), indexes(old_iterator.indexes), sliding_index(old_iterator.sliding_index) {}

    T& operator*() const {
        return ttensor(indexes);
    }

    //TODO se faccio il pointer di una reference funziona?
    T* operator->() const {
        return &(this->ttensor(this->indexes));
    }

    TensorIteratorFixed<T, rank> operator++(int) {
        //crea nuovo, incrementa me e ritorna l'altro
        TensorIteratorFixed<T, rank> new_iterator = TensorIteratorFixed<T, rank>(*this);
        increment(1);
        return new_iterator;
    }

    TensorIteratorFixed<T, rank>& operator++() {
        //incrementa me e ritorna la referenza
        increment(1);
        return this;
    }

    TensorIteratorFixed<T, rank> operator--(int) {
        //crea nuovo, decrementa me e ritorna l'altro
        TensorIteratorFixed<T, rank> new_iterator = TensorIteratorFixed<T, rank>(*this);
        increment(-1);
        return new_iterator;
    }

    TensorIteratorFixed<T, rank>& operator--() {
        //decrementa me e ritorna la referenza
        increment(-1);
        return this;
    }

    bool operator==(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) && other_iterator.indexes == indexes && other_iterator.sliding_index == sliding_index);
    }

    bool operator!=(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) && ( other_iterator.indexes != indexes || other_iterator.sliding_index != sliding_index) );
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
                check_indexes_equality(other_iterator) &&
                (indexes[sliding_index] < other_iterator.indexes) );
    }

    bool operator>(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) &&
                (sliding_index == other_iterator.sliding_index) &&
                check_indexes_equality(other_iterator) &&
                (indexes[sliding_index] > other_iterator.indexes) );
    }

    bool operator<=(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) &&
                (sliding_index == other_iterator.sliding_index) &&
                check_indexes_equality(other_iterator) &&
                (indexes[sliding_index] <= other_iterator.indexes) );
    }

    bool operator>=(const TensorIteratorFixed<T, rank>& other_iterator) const {
        return ( (&other_iterator.ttensor == &ttensor) &&
                (sliding_index == other_iterator.sliding_index) &&
                check_indexes_equality(other_iterator) &&
                (indexes[sliding_index] >= other_iterator.indexes) );
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
    size_t sliding_index={};

    void increment(const size_t& index_inc) {
        indexes[sliding_index] += index_inc;
        //controllo overflow
        assert(indexes[sliding_index] >= ttensor.widths[sliding_index]);
    }

    bool check_indexes_equality(TensorIteratorFixed<T> other_iter, size_t index_ignore){
        //bool ret = true;
        int i = 0;
        while (i < indexes.size() && (i == index_ignore || indexes[i] == other_iter.indexes[i]) ) {
            i++;
        }
        /* meno efficiente
        for (int i = 0; i < indexes.size(); i++){
            ret = ret && ( i == index_ignore || indexes[i] == other_iter.indexes[i]);
        }
        */
       return ret;
        return (i == indexes.size());
    }
};


#endif //TENSORLIB_TENSORLIB_H
