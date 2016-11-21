//
// Libor Novak
// 10/20/2016
//

#ifndef LEQSYSTEM_H
#define LEQSYSTEM_H

#include <vector>


class LEQSystem
{
public:

    LEQSystem (int idx);


    std::vector<std::vector<double>>& getA ();

    std::vector<double>& getb ();

    std::vector<double>& getx ();

    int getIdx () const;

    std::string print () const;

private:

    void _randomInit ();


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // System of linear equations in the form A*x = b
    std::vector<std::vector<double>> _A;
    std::vector<double> _b;
    // Solution
    std::vector<double> _x;

    // Identifier of the equation system
    int _idx;

};

#endif // LEQSYSTEM_H
