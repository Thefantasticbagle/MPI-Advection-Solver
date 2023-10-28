/**
 * An solver for the Diffusion Problem using MPI.
 * https://en.wikipedia.org/wiki/Diffusion_equation
 * 
 * @file diffusion_solver.c
 * @author Lars L Ruud
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <stdbool.h>

#include <mpi.h>

#include "../inc/utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;

int_t
    y_size,
    x_size,
    iterations,
    snapshot_frequency;

real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dt;

int
    world_size,
    world_rank;

MPI_Comm cartesian_comm;
int cartesian_rank,
    cartesian_size;
int cartesian_coords[2],
    cartesian_dims[2] = { 0, 0 };

int_t
    subgrid_height,
    subgrid_width,
    subgrid_area,
    subgrid_y0,
    subgrid_x0;

MPI_Datatype
    grid,
    view,
    cartesian_row,
    cartesian_column;

#define T(x,y)                      temp[0][(y) * (subgrid_width + 2) + (x)]
#define T_next(x,y)                 temp[1][((y) * (subgrid_width + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (subgrid_width + 2) + (x)]

#define MPI_BORDERS_LEFT    ( cartesian_coords[1] == 0 )
#define MPI_BORDERS_RIGHT   ( cartesian_coords[1] == cartesian_dims[1] - 1 )
#define MPI_BORDERS_TOP     ( cartesian_coords[0] == 0 )
#define MPI_BORDERS_BOTTOM  ( cartesian_coords[0] == cartesian_dims[0] - 1 )

void create_datatypes ( void );
void time_step ( void );
void boundary_condition( void );
void border_exchange( void );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );


/**
 * Swaps the values of two arrays.
 * 
 * @param m1 The first array.
 * @param m2 The second array.
*/
void
swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}


/**
 * The main function.
*/
int
main ( int argc, char **argv )
{
    // Init
    MPI_Init( NULL, NULL );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );

    // Parse and broadcast args
    if ( world_rank == 0 )
    {
        ARGS *args = parse_args( argc, argv );
        if ( !args )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit(1);
        }

        x_size = args->x_size;
        y_size = args->y_size;
        iterations = args->iterations;
        snapshot_frequency = args->snapshot_frequency;
    }

    MPI_Bcast ( &x_size, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &y_size, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &iterations, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &snapshot_frequency, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );

    // Set up and create cartesian communicator
    MPI_Dims_create( world_size, 2, cartesian_dims );

    int periods[2] = { true, true };
    int reorder = true;

    MPI_Cart_create( MPI_COMM_WORLD, 2, cartesian_dims, periods, reorder, &cartesian_comm );

    // Get our rank and coordinates within the cartesian communicator, as well as its size
    MPI_Comm_size( cartesian_comm, &cartesian_size );
    MPI_Comm_rank( cartesian_comm, &cartesian_rank );
    MPI_Cart_coords( cartesian_comm, cartesian_rank, 2, cartesian_coords );

    // Define subgrid boundaries
    subgrid_height = y_size / cartesian_dims[0];
    subgrid_width = x_size / cartesian_dims[1];
    subgrid_area = (subgrid_width + 2) * (subgrid_height + 2);
    subgrid_y0 = cartesian_coords[0] * subgrid_height;
    subgrid_x0 = cartesian_coords[1] * subgrid_width;
    printf(
        "Subgrid(%i): height = %ld, width = %ld, area = %ld, y0 = %ld, x0 = %ld, left = %i, right = %i, top = %i, bottom = %i\n", 
        cartesian_rank,
        subgrid_height,
        subgrid_width,
        subgrid_area,
        subgrid_y0,
        subgrid_x0,
        MPI_BORDERS_LEFT,
        MPI_BORDERS_RIGHT,
        MPI_BORDERS_TOP,
        MPI_BORDERS_BOTTOM
    );

    // Create datatypes and initialize domain. Take note of the time.
    create_datatypes();
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    // Iterate...
    for ( int_t iteration = 0; iteration <= iterations; iteration++ )
    {
        border_exchange();

        boundary_condition();

        time_step();

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                iterations,
                100.0 * (real_t) iteration / (real_t) iterations
            );

            domain_save ( iteration );
        }

        swap( &temp[0], &temp[1] );
    }

    // Finalize, writing the amount of time the program took and freeing allocated/created datatypes.
    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );

    domain_finalize();

    MPI_Finalize();
    exit ( EXIT_SUCCESS );
}


/**
 * Creates the necessary datatypes.
 * These should be de-allocated when no longer used.
 * 
 * @see domain_finalize()
*/
void
create_datatypes ( )
{
    // Create datatype for the entire grid, but remove the halo
    int start[2] = { 1, 1 };
    int arrsize[2] = { (subgrid_height + 2), (subgrid_width + 2) };
    int gridsize[2] = { subgrid_height, subgrid_width };

    MPI_Type_create_subarray (2, arrsize, gridsize, start, MPI_ORDER_C, MPI_DOUBLE, &grid);
    MPI_Type_commit ( &grid );

    // Create datatype for the subgrid's view of the data file
    int startV[2] = { cartesian_coords[0] * subgrid_height, cartesian_coords[1] * subgrid_width };
    int arrsizeV[2] = { y_size, x_size };
    int gridsizeV[2] = { subgrid_height, subgrid_width };

    MPI_Type_create_subarray( 2, arrsizeV, gridsizeV, startV, MPI_ORDER_C, MPI_DOUBLE, &view );
    MPI_Type_commit( &view );

    // Create row- and column datatypes for easier border exchange
    MPI_Type_create_hvector (
        subgrid_height,
        1,
        (subgrid_width + 2) * sizeof(real_t),
        MPI_DOUBLE,
        &cartesian_column
    );

    MPI_Type_contiguous (
        subgrid_width + 2,
        MPI_DOUBLE,
        &cartesian_row
    );

    MPI_Type_commit ( &cartesian_column );
    MPI_Type_commit ( &cartesian_row );
}


/**
 * Calculates a step of the differential equation.
*/
void
time_step ( void )
{
    real_t c, t, b, l, r, K, new_value;

    for ( int_t y = 1; y <= subgrid_height; y++ )
    {
        for ( int_t x = 1; x <= subgrid_width; x++ )
        {
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

            T_next(x, y) = new_value;
        }
    }
}


/**
 * Exchanges border values with neighboring processes (in the cartesian grid).
 * The cartesian_row and cartesian_column datatypes must be commited!
 * 
 * @see create_datatypes()
*/
void
border_exchange ( void )
{
    // Get left, right, up, and down ranks
    int
        cartesian_rank_left,
        cartesian_rank_right,
        cartesian_rank_up,
        cartesian_rank_down;

    MPI_Cart_shift (
        cartesian_comm,
        0,
        1,
        &cartesian_rank_left,
        &cartesian_rank_right
    );

    MPI_Cart_shift (
        cartesian_comm,
        1,
        1,
        &cartesian_rank_up,
        &cartesian_rank_down
    );

    // Send and recieve values left and right
    MPI_Sendrecv(
        &T(0,1),
        1,
        cartesian_row,
        cartesian_rank_left,
        0,

        &T(0, (subgrid_height + 1)),
        1,
        cartesian_row,
        cartesian_rank_right,
        0,
        cartesian_comm,
        MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        &T(0, subgrid_height),
        1,
        cartesian_row,
        cartesian_rank_right,
        1,

        &T(0, 0),        
        1,
        cartesian_row,
        cartesian_rank_left,
        1,
        cartesian_comm,
        MPI_STATUS_IGNORE
    );

    // Send and recieve values up and down
    MPI_Sendrecv(
        &T(1, 1),
        1,
        cartesian_column,
        cartesian_rank_up,
        0,

        &T(subgrid_width + 1, 1),
        1,
        cartesian_column,
        cartesian_rank_down,
        0,
        cartesian_comm,
        MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        &T(subgrid_width, 1),
        1,
        cartesian_column,
        cartesian_rank_down,
        1,

        &T(0, 1),
        1,
        cartesian_column,
        cartesian_rank_up,
        1,
        cartesian_comm,
        MPI_STATUS_IGNORE
    );
}


/**
 * Handles the boundary conditions.
*/
void
boundary_condition ( void )
{
    if ( MPI_BORDERS_TOP )
    {
        for ( int_t x = 1; x <= subgrid_width; x++ )
        {
            T(x, 0) = T(x, 2);
        }
    }
    if ( MPI_BORDERS_BOTTOM )
    {
        for ( int_t x = 1; x <= subgrid_width; x++ )
        {
            T(x, subgrid_height+1) = T(x, subgrid_height-1);
        }
    }
    if ( MPI_BORDERS_LEFT )
    {
        for ( int_t y = 1; y <= subgrid_height; y++ )
        {
            T(0, y) = T(2, y);
        }
    }
    if ( MPI_BORDERS_RIGHT )
    {
        for ( int_t y = 1; y <= subgrid_height; y++ )
        {
            T(subgrid_width+1, y) = T(subgrid_width-1, y);
        }
    }
}


/**
 * Initializes the domain.
 * Allocates memory for temp and thermal_diffusivity, these should be freed when no longer used.
 * 
 * @see domain_finalize()
*/
void
domain_init ( void )
{
    temp[0] = malloc ( subgrid_area * sizeof(real_t) );
    temp[1] = malloc ( subgrid_area * sizeof(real_t) );
    thermal_diffusivity = malloc ( subgrid_area * sizeof(real_t) );

    dt = 0.1;

    for ( int_t y = 1; y <= subgrid_height; y++ )
    {
        for ( int_t x = 1; x <= subgrid_width; x++ )
        {
            real_t
                temperature = 30 + 30 * sin((subgrid_x0 + x + subgrid_y0 + y) / 20.0),
                diffusivity = 0.05 + (30 + 30 * sin((x_size - (subgrid_x0 + x) + (subgrid_y0 + y)) / 20.0)) / 605.0; 

            T(x,y) = temperature;
            T_next(x,y) = temperature;
            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }
}


/**
 * Saves the information within temp to file.
 * The grid and view datatypes must be commited!

 * @see create_datatypes()
*/
void
domain_save ( int_t iteration )
{
    // Write to file
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    MPI_File out;
    MPI_File_open (
        cartesian_comm,
        filename,
        MPI_MODE_CREATE | MPI_MODE_WRONLY,
        MPI_INFO_NULL,
        &out
    );

    if ( ! out ) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }

    MPI_File_set_view( out, 0, MPI_DOUBLE, view, "native", MPI_INFO_NULL );
    MPI_File_write_all( out, &temp[0][0], 1, grid, MPI_STATUS_IGNORE );

    MPI_File_close(&out);
}


/**
 * Frees allocated memory and data types.
*/
void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
    MPI_Type_free ( &grid );
    MPI_Type_free ( &view );
    MPI_Type_free ( &cartesian_row );
    MPI_Type_free ( &cartesian_column );
}
