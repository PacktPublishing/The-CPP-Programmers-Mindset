//
// Created by sammorley on 04/01/25.
//

#ifndef CLAMP_LOOP_H
#define CLAMP_LOOP_H

#include <cstdint>
#include <span>
#include <vector>


namespace ct {

void clamp_min(std::span<uint16_t> x);
void clamp_conditional(std::span<uint16_t> x);

}


#endif //CLAMP_LOOP_H
