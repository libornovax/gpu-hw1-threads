#ifndef UTILS_H
#define UTILS_H

#include <vector>


namespace utils {

    /**
     * @brief Generates a random sequence of numbers
     * @return vector of size SEQUENCE_LENGTH
     */
    std::vector<float> generateRandomSequence ();


    /**
     * @brief Prints the sequence of numbers
     */
    void printSequence (const std::vector<float> &seq);
    void printSequence (const float *seq, int length);

    /**
     * @brief Compares two sequences and returns true if they are the same, including order of numbers
     * @param seq1
     * @param seq2
     * @return
     */
    bool compareSequences (const std::vector<float> &seq1, const std::vector<float> &seq2);

}


#endif // UTILS_H
