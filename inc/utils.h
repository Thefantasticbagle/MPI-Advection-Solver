/**
 * Utility functions header file.
 * 
 * @file utils.h
 * @author Lars L Ruud
*/

#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdint.h>


typedef int64_t int_t;

typedef struct args_struct {
    int_t y_size;
    int_t x_size;
    int_t iterations;
    int_t snapshot_frequency;
} ARGS;

ARGS *parse_args ( int argc, char **argv );
void help ( char const *exec, char const opt, char const *optarg );

#endif
