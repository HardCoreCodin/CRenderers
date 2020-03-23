#pragma once
#include "lib/core/types.h"

#define Kilobytes(value) ((value)*1024LL)
#define Megabytes(value) (Kilobytes(value)*1024LL)
#define Gigabytes(value) (Megabytes(value)*1024LL)
#define Terabytes(value) (Gigabytes(value)*1024LL)

typedef struct Memory {
    u8* address;
    u64 occupied;
} Memory;

void* allocate(Memory* memory, u64 size) {
    memory->occupied += size;

    void* address = memory->address;
    memory->address += size;
    return address;
}