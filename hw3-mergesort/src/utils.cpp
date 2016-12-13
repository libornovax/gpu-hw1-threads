#include "utils.h"

#include <cassert>
#include <random>
#include <iostream>
#include "settings.h"


namespace utils {

    std::vector<float> generateRandomSequence ()
    {
        std::vector<float> seq(SEQUENCE_LENGTH);

        std::random_device rd;
        std::mt19937 mt(rd());

        std::uniform_real_distribution<float> dist(0, 100);

        // Fill the sequence with random values
        for (auto &num: seq) num = dist(mt);

        return seq;
    }


    void printSequence (const std::vector<float> &seq)
    {
        for (auto &num: seq) std::cout << num << " ";
        std::cout << std::endl;
    }


    void printSequence (const float *seq, int length)
    {
        for (int i = 0; i < length; ++i) std::cout << seq[i] << " ";
        std::cout << std::endl;
    }


    bool compareSequences (const std::vector<float> &seq1, const std::vector<float> &seq2)
    {
        assert(seq1.size() == seq2.size());

        for (int i = 0; i < seq1.size(); ++i)
        {
            if (seq1[i] != seq2[i]) return false;
        }

        return true;
    }

}

