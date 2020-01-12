#include "sequences.h"
#include "constants.h"

int main() {
    switch (MOTION_TYPE) {
        case 1:
            sequenceFromKittiOpticalFlow();
            break;
        case 2:
            sequenceFromKitti2D2D();
            break;
        case 3:
            sequenceFromKitti3D2D();
            break;
        case 4:
            sequenceFromKitti3D3D();
            break;
        default:
            std::cerr << "Wrong Motion Option" << std::endl;
            exit(EXIT_FAILURE);
    }
}

