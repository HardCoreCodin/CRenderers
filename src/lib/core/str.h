#pragma once
#include "lib/core/types.h"

void printNumberIntoString(u16 number, char* str) {
    char *string = str;
    if (number) {
        u16 temp;
        temp = number;
        number /= 10;
        *string-- = (char)('0' + temp - number * 10);

        if (number) {
            temp = number;
            number /= 10;
            *string-- = (char)('0' + temp - number * 10);

            if (number) {
                temp = number;
                number /= 10;
                *string-- = (char)('0' + temp - number * 10);

                if (number) {
                    temp = number;
                    number /= 10;
                    *string = (char)('0' + temp - number * 10);
                } else
                    *string = ' ';
            } else {
                *string-- = ' ';
                *string-- = ' ';
            }
        } else {
            *string-- = ' ';
            *string-- = ' ';
            *string-- = ' ';
        }
    } else {
        *string-- = '0';
        *string-- = ' ';
        *string-- = ' ';
        *string   = ' ';
    }
}