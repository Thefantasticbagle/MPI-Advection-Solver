# mpi-diffusion-solver
A parallel implementation of the [Finite Difference Method (FDM)](https://en.wikipedia.org/wiki/Finite_difference_method) for solving the 2D heat problem, made using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) in [C](https://en.wikipedia.org/wiki/C_(programming_language)).

![The diffusion solver in action](https://github.com/Thefantasticbagle/pthreads-diffusion-solver/blob/129ace32607ceceb2827cc3cab52a362b476c1ca/example_animation.gif)

## Setup
### Downloading the repository
```sh
$ git clone https://github.com/Thefantasticbagle/mpi-diffusion-solver.git
$ cd mpi-diffusion-solver
```

### Installing dependencies
**OpenMPI**

Linux/Ubuntu:
```sh
$ sudo apt update
$ sudo apt install -y openmpi-bin openmpi-doc libopenmpi-dev
```

MacOSX:
```sh
$ brew update
$ brew install open-mpi
```

**gnuplot**

Linux/Ubuntu:
```sh
$ sudo apt update
$ sudo apt install gnuplot
```

MacOSX:
```sh
$ brew update
$ brew install gnuplot
```

**ffmpeg**

Linux/Ubuntu:

```sh
$ sudo apt update
$ sudo apt install ffmpeg
```

MacOSX:

```sh
$ brew update
$ brew install ffmpeg
```

### Setting up folder structure
```sh
$ make setup 
```
Creates the folders `data`, `plots`, and `video` which are necessary to run the program. These can be emptied with `make purge`.

## Run
```sh
$ make diffusion_solver
```
Compiles the program. To delete it, use `make clean`.

```sh
$ mpirun -np 4 ./diffusion_solver
```
Runs the program with default arguments.

```sh
$ make video
```
Creates `plots` for the previous run and a `video` animation of them, which is located at `./video/animation.mp4` by default.

## Options
`mpirun -np [number_processes] ./diffusion_solver -y [y_size] -x [x_size] -i [iterations] -s [snapshot_freq]`
Option | Description | Restrictions | Default value
:------------ | :------------ | :------------ | :------------
**-y** | Height in pixels | > 0 | 256
**-x** | Width in pixels | > 0 | 256
**-i** | Number of iterations | > 0 | 100000
**-s** | Number of iterations between each snapshot | > 0 | 1000
**-np**| MPI number of processes | > 0 | 4