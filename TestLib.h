//
// Created by rr on 08/11/2019.
//

#ifndef TENSORLIB_TESTLIB_H
#define TENSORLIB_TESTLIB_H

class Test{
private:
    vector<function<void()>> _functions;
    vector<string> _names;

public:
    void launch_test(int x){
        if(x == -1){
            for(unsigned long i=0; i<_functions.size();++i){
                try {
                    cout << "--------------------------------------" << endl;
                    _functions[i]();
                    cout<< endl << "concluso test "<< _names[i] <<endl;
                }catch(...) {
                    cout << "Errore al test "<< _names[i] <<endl;
                }
            }
        }else{
            try {
                cout << "--------------------------------------" << endl;
                _functions[x]();
                cout<< endl << "concluso test "<< _names[x] <<endl;
            }catch(...) {
                cout << "Errore al test "<< _names[x] <<endl;
            }
        }

    }
    void add(const function<void()>& a, const string& name){
        _functions.push_back(a);
        _names.push_back(name);
    }
};

#endif //TENSORLIB_TESTLIB_H
