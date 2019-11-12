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
                    cout << "vvvv---------------TEST " << i << "-----------------------vvvv" << endl;
                    _functions[i]();
                    cout<< endl << "***Concluso test ["<< _names[i] << "]" <<endl;
                    cout << "^^^^---------------TEST " << i << "-----------------------^^^^" << endl;
                }catch(...) {
                    cout << "***Errore al test ["<< _names[x] << "]" <<endl;
                }
            }
        }else{
            try {
                cout << "VVVV---------------TEST " << x << "-----------------------VVVV" << endl;
                _functions[x]();
                cout<< endl << "***Concluso test ["<< _names[x] << "]" <<endl;
                cout << "^^^^---------------TEST " << x << "-----------------------^^^^" << endl;
            }catch(...) {
                cout << "***Errore al test ["<< _names[x] << "]" <<endl;
            }
        }

    }
    void add(const function<void()>& a, const string& name){
        _functions.push_back(a);
        _names.push_back(name);
    }
};

#endif //TENSORLIB_TESTLIB_H
