#pragma once
#include "lib/core/types.h"

#define Kilobytes(value) ((value)*1024LL)
#define Megabytes(value) (Kilobytes(value)*1024LL)
#define Gigabytes(value) (Megabytes(value)*1024LL)
#define Terabytes(value) (Gigabytes(value)*1024LL)
#define Alloc(T) (T*)allocate(sizeof(T))
#define AllocN(T, N) (T*)allocate(sizeof(T) * N)

#define MEMORY_SIZE Gigabytes(1)
#define MEMORY_BASE Terabytes(2)

typedef struct Memory {
    u8* address;
    u64 occupied;
} Memory;
static Memory memory;

void* allocate(u64 size) {
    memory.occupied += size;

    void* address = memory.address;
    memory.address += size;
    return address;
}