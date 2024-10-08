#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "decs.h"
#include "harm_model.h"

void init_harm_data(char *fname)
{
    FILE *fp;
    double x[4];
    double rp, hp, V, dV, two_temp_gam;
    int i, j, k;

    /* header variables not used except locally */
    double t, tf, cour, DTd, DTl, DTi, dt;
    int nstep, DTr, dump_cnt, image_cnt, rdump_cnt, lim, failed;
    double r, h, divb, vmin, vmax, gdet;
    double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];

    fp = fopen(fname, "r");

    if (fp == NULL) {
        fprintf(stderr, "can't open sim data file\n");
        exit(1);
    } else {
        fprintf(stderr, "successfully opened %s\n", fname);
    }

    /* get standard HARM header */
    if (fscanf(fp, "%lf", &t) != 1 ||
        fscanf(fp, "%d", &N1) != 1 ||
        fscanf(fp, "%d", &N2) != 1 ||
        fscanf(fp, "%lf", &startx[1]) != 1 ||
        fscanf(fp, "%lf", &startx[2]) != 1 ||
        fscanf(fp, "%lf", &dx[1]) != 1 ||
        fscanf(fp, "%lf", &dx[2]) != 1 ||
        fscanf(fp, "%lf", &tf) != 1 ||
        fscanf(fp, "%d", &nstep) != 1 ||
        fscanf(fp, "%lf", &a) != 1 ||
        fscanf(fp, "%lf", &gam) != 1 ||
        fscanf(fp, "%lf", &cour) != 1 ||
        fscanf(fp, "%lf", &DTd) != 1 ||
        fscanf(fp, "%lf", &DTl) != 1 ||
        fscanf(fp, "%lf", &DTi) != 1 ||
        fscanf(fp, "%d", &DTr) != 1 ||
        fscanf(fp, "%d", &dump_cnt) != 1 ||
        fscanf(fp, "%d", &image_cnt) != 1 ||
        fscanf(fp, "%d", &rdump_cnt) != 1 ||
        fscanf(fp, "%lf", &dt) != 1 ||
        fscanf(fp, "%d", &lim) != 1 ||
        fscanf(fp, "%d", &failed) != 1 ||
        fscanf(fp, "%lf", &Rin) != 1 ||
        fscanf(fp, "%lf", &Rout) != 1 ||
        fscanf(fp, "%lf", &hslope) != 1 ||
        fscanf(fp, "%lf", &R0) != 1) {
        fprintf(stderr, "Error reading HARM header\n");
        fclose(fp);
        exit(1);
    }

    N3 = 1;
    fprintf(stderr, "Resolution: %d, %d, %d\n", N1, N2, N3);
    fprintf(stderr, "hslope = %le\n", hslope);

    startx[0] = 0.;
    startx[3] = 0.;
    stopx[0] = 1.;
    stopx[1] = startx[1] + N1 * dx[1];
    stopx[2] = startx[2] + N2 * dx[2];
    stopx[3] = 2. * M_PI;

    fprintf(stderr, "Sim range x1, x2:  %g %g, %g %g\n", startx[1], stopx[1], startx[2], stopx[2]);

    dx[0] = 1.;
    dx[3] = 2. * M_PI;

    /* Allocate storage for all model size dependent variables */
    init_storage();

    two_temp_gam = 0.5 * ((1. + 2. / 3. * (TP_OVER_TE + 1.) / (TP_OVER_TE + 2.)) + gam);
    Thetae_unit = (two_temp_gam - 1.) * (MP / ME) / (1. + TP_OVER_TE);
    dMact = 0.;
    Ladv = 0.;
    bias_norm = 0.;
    V = 0.;
    dV = dx[1] * dx[2] * dx[3];

    for (k = 0; k < N1 * N2; k++) {
        j = k % N2;
        i = (k - j) / N2;

        if (fscanf(fp, "%lf %lf %lf %lf", &x[1], &x[2], &r, &h) != 4) {
            fprintf(stderr, "Error reading coordinates\n");
            fclose(fp);
            exit(1);
        }

        bl_coord(x, &rp, &hp);
        if (fabs(rp - r) > 1.e-5 * rp || fabs(hp - h) > 1.e-5) {
            fprintf(stderr, "grid setup error\n");
            fprintf(stderr, "rp,r,hp,h: %g %g %g %g\n", rp, r, hp, h);
            fclose(fp);
            exit(1);
        }

        if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf", 
                   &p[NPRIM_INDEX(KRHO,k)],
                   &p[NPRIM_INDEX(UU,k)],
                   &p[NPRIM_INDEX(U1,k)],
                   &p[NPRIM_INDEX(U2,k)],
                   &p[NPRIM_INDEX(U3,k)],
                   &p[NPRIM_INDEX(B1,k)],
                   &p[NPRIM_INDEX(B2,k)],
                   &p[NPRIM_INDEX(B3,k)]) != 8) {
            fprintf(stderr, "Error reading primitive variables\n");
            fclose(fp);
            exit(1);
        }

        if (fscanf(fp, "%lf", &divb) != 1 ||
            fscanf(fp, "%lf %lf %lf %lf", &Ucon[0], &Ucon[1], &Ucon[2], &Ucon[3]) != 4 ||
            fscanf(fp, "%lf %lf %lf %lf", &Ucov[0], &Ucov[1], &Ucov[2], &Ucov[3]) != 4 ||
            fscanf(fp, "%lf %lf %lf %lf", &Bcon[0], &Bcon[1], &Bcon[2], &Bcon[3]) != 4 ||
            fscanf(fp, "%lf %lf %lf %lf", &Bcov[0], &Bcov[1], &Bcov[2], &Bcov[3]) != 4 ||
            fscanf(fp, "%lf", &vmin) != 1 ||
            fscanf(fp, "%lf", &vmax) != 1 ||
            fscanf(fp, "%lf", &vmin) != 1 ||
            fscanf(fp, "%lf", &vmax) != 1 ||
            fscanf(fp, "%lf", &gdet) != 1) {
            fprintf(stderr, "Error reading other data\n");
            fclose(fp);
            exit(1);
        }

        bias_norm += dV * gdet * pow(p[NPRIM_INDEX(UU,k)] / p[NPRIM_INDEX(KRHO,k)] * Thetae_unit, 2.);
        V += dV * gdet;

        if (i <= 20)
            dMact += gdet * p[NPRIM_INDEX(KRHO,k)] * Ucon[1];
        if (i >= 20 && i < 40)
            Ladv += gdet * p[NPRIM_INDEX(UU,k)] * Ucon[1] * Ucov[0];
    }

    bias_norm /= V;
    dMact *= dx[3] * dx[2];
    dMact /= 21.;
    Ladv *= dx[3] * dx[2];
    Ladv /= 21.;
    fprintf(stderr, "dMact: %g, Ladv: %g\n", dMact, Ladv);

    fclose(fp);
}
