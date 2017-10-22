

#/***********************************************************************************
#    Copyright 2013 Joshua C. Dolence, Charles F. Gammie, Monika Mo\'scibrodzka,
#                   and Po Kin Leung
#
#                        GRMONTY  version 1.0   (released February 1, 2013)
#
#    This file is part of GRMONTY.  GRMONTY v1.0 is a program that calculates the
#    emergent spectrum from a model using a Monte Carlo technique.
#
#    This version of GRMONTY is configured to use input files from the HARM code
#    available on the same site.   It assumes that the source is a plasma near a
#    black hole described by Kerr-Schild coordinates that radiates via thermal
#    synchrotron and inverse compton scattering.
#
#    You are morally obligated to cite the following paper in any
#    scientific literature that results from use of any part of GRMONTY:
#
#    Dolence, J.C., Gammie, C.F., Mo\'scibrodzka, M., \& Leung, P.-K. 2009,
#        Astrophysical Journal Supplement, 184, 387
#
#    Further, we strongly encourage you to obtain the latest version of
#    GRMONTY directly from our distribution website:
#    http://rainman.astro.illinois.edu/codelib/
#
#    GRMONTY is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    GRMONTY is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with GRMONTY; if not, write to the Free Software
#    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
#***********************************************************************************/
#
# requires an openmp-enabled version of gcc
#
CUDAC = nvcc
CUDAC_FLAGS = -Wno-deprecated-gpu-targets -O2
CC = gcc-5
CFLAGS = -Wall -O2 -fopenmp
LDFLAGS = -lm -lgsl -lgslcblas

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

ALL_CFLAGS :=
ALL_CFLAGS += $(ALL_CFLAGS)
ALL_CFLAGS += $(addprefix -Xcompiler ,$(CFLAGS))

ALL_CUDACFLAGS :=
ALL_CUDACFLAGS += $(ALL_CUDACFLAGS)
ALL_CUDACFLAGS += $(CUDAC_FLAGS)
ALL_CUDACFLAGS += -ccbin $(CC)

SRCS = grmonty.c compton.c init_geometry.c tetrads.c geodesics.c \
radiation.c jnu_mixed.c hotcross.c track_super_photon.c \
scatter_super_photon.c harm_model.c harm_utils.c init_harm_data.c

OBJS = grmonty.o compton.o init_geometry.o tetrads.o geodesics.o \
radiation.o jnu_mixed.o hotcross.o track_super_photon.o \
scatter_super_photon.o harm_model.o harm_utils.o init_harm_data.o

INCS = decs.h constants.h harm_model.h

grmonty : $(OBJS) $(INCS) makefile
	$(CC) $(CFLAGS) -o grmonty $(OBJS) $(LDFLAGS)
	# $(CUDAC) $(ALL_CUDACFLAGS) $(ALL_CFLAGS) -o grmonty $(OBJS) $(ALL_LDFLAGS)

$(OBJS) : $(INCS) makefile

clean:
	/bin/rm *.o
