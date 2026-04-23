The TSPLib parser includes a header file, which is referenced for the main.  The parser cannot be called seperately

For future testing simply include tsplib_parser.h and use the following function syntax

TspMatrixInstance inst = load_tsplib_matrix("berlin52.tsp");

Function output:

inst.name = "simple5";
inst.type = "TSP";
inst.dimension = 5;

inst.dist = {
    0,  2,  9, 10,  7,
    2,  0,  6,  4,  3,
    9,  6,  0,  8,  5,
   10,  4,  8,  0,  6,
    7,  3,  5,  6,  0
};

Usage:
g++ -std=c++17 ParsertestMain.cpp tsplib_parser.cpp -o tsp_test
./tsp_test <file.tsp>
./tsp_test berlin52.tsp

Expected output:
NAME: berlin52
TYPE: TSP
DIMENSION: 52
First 15 matrix elements:
0 666 281 396 291 326 641 427 600 1043 551 526 387 338 393

Greedy nearest-neighbor result:
Tour length: 8980

Tour (0-based indices):
0 48 31 44 18 40 7 8 9 42 ... 0