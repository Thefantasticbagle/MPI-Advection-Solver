/**
 * Utility functions.
 * 
 * @file utils.c
 * @author Lars L Ruud
*/

#include "../inc/utils.h"
#include <getopt.h>
#include <stddef.h>
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>


/**
 * Parses the program arguments.
 * If they are invalid, outputs the help screen.
 * 
 * @see help(...)
*/
ARGS
*parse_args ( int argc, char **argv )
{
    int_t y_size = 256;
    int_t x_size = 256;
    int_t iterations = 100000;
    int_t snapshot_frequency = 1000;

    static struct option const options_long[] = {
        {"help",                no_argument,       0, 'h'},
        {"y_size",              required_argument, 0, 'y'},
        {"x_size",              required_argument, 0, 'x'},
        {"iterations",          required_argument, 0, 'i'},
        {"snapshot_frequency",  required_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    static char const *options_short = "hy:x:i:s:";
    {
        char *endptr;
        int c, option_index = 0;

        while ( (c = getopt_long( argc, argv, options_short, options_long, &option_index )) != -1 )
        {
            switch (c)
            {
                case 'h':
                    help( argv[0], 0, NULL );
                    exit(0);
                    break;

                case 'y':
                    y_size = strtol(optarg, &endptr, 10);
                    if ( endptr == optarg || y_size < 0 )
                    {
                        help( argv[0], c, optarg );
                        return NULL;
                    }
                    break;

                case 'x':
                    x_size = strtol(optarg, &endptr, 10);
                    if ( endptr == optarg || x_size < 0 )
                    {
                        help( argv[0], c, optarg );
                        return NULL;
                    }
                    break;

                case 'i':
                    iterations = strtol(optarg, &endptr, 10);
                    if ( endptr == optarg || iterations < 0 )
                    {
                        help( argv[0], c, optarg);
                        return NULL;
                    }
                    break;

                case 's':
                    snapshot_frequency = strtol(optarg, &endptr, 10);
                    if ( endptr == optarg || snapshot_frequency < 0 )
                    {
                        help( argv[0], c, optarg );
                        return NULL;
                    }
                    break;

                default:
                    abort();
            }
        }
    }

    if ( argc < (optind) )
    {
        help( argv[0], ' ', "Not enough arugments" );
        return NULL;
    }

  ARGS* args_parsed = (ARGS*) malloc( sizeof(ARGS) );
  args_parsed->y_size = y_size;
  args_parsed->x_size = x_size;
  args_parsed->iterations = iterations;
  args_parsed->snapshot_frequency = snapshot_frequency;

  return args_parsed;
}


/**
 * Outputs the help screen.
*/
void
help ( char const *exec, char const opt, char const *optarg )
{
    FILE *out = stdout;

    if ( opt != 0 )
    {
        out = stderr;
        if ( optarg )   fprintf(out, "Invalid parameter: %c %s\n", opt, optarg);
        else            fprintf(out, "Invalid parameter: %c\n", opt);
    }

    fprintf(out, "%s [options]\n\n", exec);

    fprintf(out, "Options                   Description                     Restriction     Default\n");
    fprintf(out, "  -y, --y_size            height of the grid              n>0             256\n"    );
    fprintf(out, "  -x, --x_size            width of the grid               n>0             256\n"    );
    fprintf(out, "  -i, --iterations        number of iterations            i>0             100000\n" );
    fprintf(out, "  -s, --snapshot_freq     snapshot frequency              s>0             1000\n\n"  );

    fprintf(out, "Example: %s -y 512 -x 512 -i 100000 -s 1000\n", exec);
}