#ifndef LEQSYSTEM_H
#define LEQSYSTEM_H

#include <vector>


class LEQSystem
{
public:

    LEQSystem (int idx);


    std::vector<std::vector<double>>& getA ();

    std::vector<double>& getb ();

    int getIdx () const;


private:

    void _randomInit ();


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // System of linear equations in the form A*x = b
    std::vector<std::vector<double>> _A;
    std::vector<double> _b;

    // Identifier of the equation system
    int _idx;

};

#endif // LEQSYSTEM_H
