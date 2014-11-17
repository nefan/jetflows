/*
 * This file is part of jetflows.
 *
 * Copyright (C) 2014, Henry O. Jacobs (hoj201@gmail.com), Stefan Sommer (sommer@di.ku.dk)
 * https://github.com/nefan/jetflows.git
 *
 * jetflows is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * jetflows is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with jetflows.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef GAUSSIANKERNELS_H
#define GAUSSIANKERNELS_H

#include <omp.h>
#include "math.h"
typedef double scalar;
#include "sla.h"

inline double Gaussian_monomial(double xdSIGMA, int n) {
    double y = xdSIGMA * exp( -(0.5/n) * pow(xdSIGMA,2) );
    return pow(y,n);
}

inline void diff_1D_Gaussian_parallel_cpp(double* in, int inn, double* out, int outn, const int k, double SIGMA, bool parallel) {

    assert(inn = outn);

    int nrThreads = 1;
    if (parallel)
        nrThreads = omp_get_max_threads();

#pragma omp parallel for schedule(static) shared(in,out) num_threads(nrThreads)
    for (int i=0; i<inn; i++) {
        double x = in[i];
        double xdSIGMA = x/SIGMA;
        double G = exp(-.5*pow(xdSIGMA,2));

        switch (k) {
            case 0:
                out[i] = G;
                break;
            case 1:
                out[i] = -1.*Gaussian_monomial(xdSIGMA,1) / SIGMA;
                break;
            case 2:
                out[i] =  ( Gaussian_monomial(xdSIGMA,2) - G ) / pow(SIGMA,2);
                break;
            case 3:
                out[i] = -1.*( Gaussian_monomial(xdSIGMA,3) - 3.*Gaussian_monomial(xdSIGMA,1)) / pow(SIGMA,3);
                break;
            case 4:
                out[i] = (Gaussian_monomial(xdSIGMA,4) - 6.*Gaussian_monomial(xdSIGMA,2) + 3.*G ) / pow(SIGMA,4);
                break;
            case 5:
                out[i] = (-1.*(Gaussian_monomial(xdSIGMA,5) - 10.*Gaussian_monomial(xdSIGMA,3) + 15.*Gaussian_monomial(xdSIGMA,1) )) / pow(SIGMA,5);
                break;
            case 6:
                out[i] = (Gaussian_monomial(xdSIGMA,6) - 15.*Gaussian_monomial(xdSIGMA,4) + 45.*Gaussian_monomial(xdSIGMA,2) -15.*G) / pow(SIGMA,6);
                break;
        }
    }
}

#endif // GAUSSIANKERNELS_H

