PROGNAME:=diffusion_solver

DATADIR:=data/
PLOTSDIR:=plots/
VIDEODIR:=video/

DATA:=$(DATADIR)*.bin
PLOTS:=$(PLOTSDIR)*.png
VIDEO:=$(VIDEODIR)*.mp4

CC:=mpicc
CFLAGS+=-std=c11
LDLIBS+=-lm

SRC:=src/$(PROGNAME).c src/utils.c

.PHONY: clean purge setup video

diffusion_solver: $(SRC)
	$(CC) $^ $(CFLAGS) -o $@ $(LDLIBS)

clean:
	-rm -f diffusion_solver

purge:
	-rm -f diffusion_solver $(DATA) $(PLOTS) $(VIDEO)

setup:
	-mkdir -p data plots video

video:
	./plot.sh > /dev/null
	ffmpeg -y -i $(PLOTSDIR)%5d.png -vf format=yuv420p $(VIDEODIR)animation.mp4 &> /dev/null