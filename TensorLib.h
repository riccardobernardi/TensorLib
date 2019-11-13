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


using namespace std;

// TODO controlli sulla larghezza del vettore
template<class T = int, int rank=-1>
class Tensor {
public:
    // when the rank is not specified
    Tensor<T,rank>(std::initializer_list<size_t>&& a){
        // cout << "costruttore : Tensor<T>(std::initializer_list<size_t>&& a)" << endl;
        //TODO decidere se rank può essere 0
        assert(a.size()==rank);

        cout << "sto usando il generico con valore rank:" << rank << endl;

        widths = a;
        strides = cummult(widths);
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    // when the rank is specified
/*
    Tensor<T,rank>(const std::vector<size_t>& a){
        // cout << "costruttore : Tensor<T>(std::vector<size_t>& a)" << endl;
        assert(a.size()==rank);
        widths = a;
        strides = cummult(widths);
    }
*/

    // copy constructor
    Tensor<T, rank>(const Tensor<T, rank>& a){
        widths = a.widths;
        strides = a.strides;
        data = a.data;
        offset = a.offset;
    }

    //move constructor, se forniamo il move constructor dobbiamo permettere di reinserire il vettore di dati, infatti il tensore che viene passato come parametro al move constructor rimarrà vuoto.
    Tensor<T, rank>(Tensor<T, rank>&& a){
        widths = a.widths;
        strides = a.strides;
        data = a.data;
        offset = a.offset;

        a.widths = std::vector<size_t>();
        a.strides = std::vector<size_t>();
        a.data = std::shared_ptr<std::vector<T>>();             //non è necessario fare altre operazioni sul vecchio shared_pointer (per fare in modo che decrementi il contatore di pointers attivi)
        a.offset = 0;                                           //poichè l'operatore = è overloadato e ci pensano loro
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

        assert(data.get() != nullptr);
        return (data.get()->at(tmp));
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
            //TODO caso in cui il risultante è un tensore di grado 0 o 1
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

    //default constructor
    Tensor<T, rank>() : widths(), strides(), data() {      //lo usiamo internamente per comodità (nelle slice/window), non ha senso di essere pubblico
        offset = 0;
    }

};

// tutti i tensori con dimensione non specificata staticamente vanno qui
// ha gli stessi metodi di quello sopra
template<class T>
class Tensor<T,-1> {
public:
    // when the rank is not specified
    Tensor<T>(std::initializer_list<size_t>&& a){
        // cout << "costruttore : Tensor<T>(std::initializer_list<size_t>&& a)" << endl;
        //TODO decidere se rank può essere 0

        cout << "stiamop usando la spec" << endl;


        widths = a;
        strides = cummult(widths);
        cout <<"www"<< widths[0] << endl;
        cout <<"sss"<< strides[0] <<endl;
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    // when the rank is specified
    Tensor<T>(const std::vector<size_t>& a){
        // cout << "costruttore : Tensor<T>(std::vector<size_t>& a)" << endl;

        widths = a;
        strides = cummult(widths);
    }

    // copy constructor
    Tensor<T>(const Tensor<T>& a){
        widths = a.widths;
        strides = a.strides;
        data = a.data;
        offset = a.offset;
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

    // initialize with an array that will be represented as a tensor
    void initialize(std::initializer_list<T>&& a){
        if ( (data.get() == nullptr) || (a.size() == data.get()->size()) ){
            data = make_shared<std::vector<T>>(a);
        }else{
            // cout << a.size() << data.get()->size() << endl;
            cout << "Una volta inizializzato il vettore non può essere modificato nelle dimensioni!" << endl;
        }
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

// tensori caso speciale di dimensione 0, serve per la slice
// non ha metodi se non rank() che torna 0
template<class T>
class Tensor<T,0> {
public:

    template<typename, typename> friend class Tensor;

    size_t rank(){
        return 0;
    }

private:

    // metadata : immutable
    std::vector<size_t> widths;
    std::vector<size_t> strides;
    size_t offset=0;

    //data : mutable
    std::shared_ptr<std::vector<T>> data;
};

// tensore di una dimensione, alias vettore, serve per la slice
// non ha la slice e la flatten
template<class T>
class Tensor<T,1> {
public:

    template<typename, typename> friend class Tensor;

    // when the rank is not specified
    Tensor<T>(std::initializer_list<size_t>&& a){
        // cout << "costruttore : Tensor<T>(std::initializer_list<size_t>&& a)" << endl;
        //TODO decidere se rank può essere 0

        cout << "stiamop usando la spec" << endl;


        widths = a;
        strides = cummult(widths);
        cout <<"www"<< widths[0] << endl;
        cout <<"sss"<< strides[0] <<endl;
        data = std::make_shared<std::vector<T>>(strides[0] * widths[0], 0); //vettore lungo mult(width) di zeri
    }

    // when the rank is specified
    Tensor<T>(const std::vector<size_t>& a){
        // cout << "costruttore : Tensor<T>(std::vector<size_t>& a)" << endl;

        widths = a;
        strides = cummult(widths);
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
            // cout << "stride: " << strides[i] << endl;
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

template<class T>
class TensorIterator {
public:

    TensorIterator<T>(const Tensor<T>& tensor) {
        this->tensor = tensor;
        indexes = std::vector<size_t>(this->tensor.widths.size(), 0);
    }

    TensorIterator<T>(const TensorIterator<T>& old_iterator) {
        this->tensor = old_iterator.tensor;
        this->indexes = old_iterator.indexes;
    }

    T& operator*() const {
        return this->tensor(this->indexes);
    }

    //TODO se faccio il pointer di una reference funziona?
    T* operator->() const {
        return &(this->tensor(this->indexes));
    }

    TensorIterator<T> operator++(int) {
        //TODO crea nuovo, incrementa me e ritorna l'altro
        TensorIterator<T> new_iterator = TensorIterator<T>(*this);
        increment(1);
        return new_iterator;
    }

    TensorIterator<T>& operator++() {
        //TODO incrementa me e ritorna la referenza
        increment(1);
        return *this;
    }

    bool operator==(const TensorIterator<T>& other_iterator) const {
        return (other_iterator.tensor == tensor && other_iterator.indexes == indexes);
    }

    bool operator!=(const TensorIterator<T>& other_iterator) const {
        return (other_iterator.tensor != tensor || other_iterator.indexes != indexes);
    }

    TensorIterator<T>& operator+=(int inc) {
        if (inc >= 0 ){
            this.increment(inc);
        } else {
            //this.decrement(-inc)
        }

        return *this;
    }

    TensorIterator<T>& operator-=(int dec) {
        if (dec <= 0 ){
            this.increment(-dec);
        } else {
            //this.decrement(dec)
        }

        return *this;
    }

    TensorIterator<T> operator+(int inc) {
        TensorIterator<T> new_iterator = TensorIterator<T>(*this);
        if (inc >= 0 ){
            new_iterator.increment(inc);
        } else {
            //new_iterator.decrement(-inc)
        }

        return new_iterator;
    }

    TensorIterator<T> operator-(int dec) {
        TensorIterator<T> new_iterator = TensorIterator<T>(*this);
        if (dec < 0 ){
            new_iterator.increment(-inc);
        } else {
            //new_iterator.decrement(inc)
        }

        return new_iterator;
    }

    //TODO <, >, <=, >=, decrement, iter - iter, direct access[], (n+iter)



private:
    Tensor<T>& tensor;
    std::vector<size_t> indexes;

    void increment(const size_t& index_inc) {
        size_t last_index = this->indexes.size() - 1;
        this->indexes[last_index] += index_inc;
        for (size_t i = last_index; i > 0; i--) {
            //controllare che faccia divisione intera
            this->indexes[i-1] += ( this->indexes[i] / this->tensor.widths[i]);
            this->indexes[i] = ( this->indexes[i] % this->tensor.widths[i]);
        }
        //controllo overflow
        assert(this->indexes[0] >= this->tensor.widths[0]);
    }

};

template<class T>
class TensorIteratorFixed{
public:

    TensorIterator<T>(const Tensor<T>& tensor, const std::vector<size_t>& starting_indexes, const size_t& sliding_index) {
        size_t indexes_size = starting_indexes.size();
        assert(indexes_size == tensor.widths.size());

        assert(sliding_index >= 0 && sliding_index < indexes_size);
        for (size_t i = 0; i < indexes_size; i++){
            assert(starting_indexes[i] < tensor.widths[i] );
        }

        this->tensor = tensor;
        indexes = std::vector<size_t>(this->tensor.widths.size(), 0);
        this->sliding_index = sliding_index;
        //TODO per adesso assumiamo che sullo sliding_index si parta da 0
        this->index[sliding_index] = 0;
    }

    TensorIterator<T>(const TensorIterator<T>& old_iterator) {
        this->tensor = old_iterator.tensor;
        this->indexes = old_iterator.indexes;
        this->sliding_index = old_iterator.sliding_index;
    }

    T& operator*() const {
        return this->tensor(this->indexes);
    }

    //TODO se faccio il pointer di una reference funziona?
    T* operator->() const {
        return &(this->tensor(this->indexes));
    }

    TensorIterator<T> operator++(int) {
        //TODO crea nuovo, incrementa me e ritorna l'altro
        TensorIterator<T> new_iterator = TensorIterator<T>(*this);
        increment(1);
        return new_iterator;
    }

    TensorIterator<T>& operator++() {
        //TODO incrementa me e ritorna la referenza
        increment(1);
        return this;
    }

    bool operator==(const TensorIterator<T>& other_iterator) const {
        return (other_iterator.tensor == tensor && other_iterator.indexes == indexes && other_iterator.sliding_index == sliding_index);
    }

    bool operator!=(const TensorIterator<T>& other_iterator) const {
        return (other_iterator.tensor != tensor || other_iterator.indexes != indexes || other_iterator.sliding_index != sliding_index);
    }


private:

    Tensor<T>& tensor;
    std::vector<size_t> indexes;
    size_t sliding_index;

    void increment(const size_t& index_inc) {
        this->indexes[sliding_index] += index_inc;
        //controllo overflow
        assert(this->indexes[sliding_index] >= this->tensor.widths[sliding_index]);
    }
}

#endif //TENSORLIB_TENSORLIB_H
