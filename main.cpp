#include<iostream>
using namespace std;

pair<int, int> botak(){
    return {2, 3};
}

int main(){
    cout << botak()[0] << endl;
}