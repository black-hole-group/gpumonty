/*
 * GPUmonty - constants.h
 * Copyright (C) 2026 Pedro Naethe Motta
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
 */

/* == Definition of constants in CGS units == */

/** Electron charge */
#define EE              (4.80320680e-10) 
/** Speed of light */
#define CL              (2.99792458e10) 
/** Electron mass */
#define ME              (9.1093826e-28) 
/** Proton mass */
#define MP              (1.67262171e-24) 
/** Neutron mass */
#define MN              (1.67492728e-24) 
/** Atomic mass unit */
#define AMU             (1.66053886e-24) 
/** Planck constant */
#define HPL             (6.6260693e-27) 
/** Reduced Planck constant (h-bar) */
#define HBAR            (HPL/(2.*M_PI)) 
/** Boltzmann constant */
#define KBOL            (1.3806505e-16) 
/** Gravitational constant */
#define GNEWT           (6.6742e-8) 
/** Stefan-Boltzmann constant */
#define SIG             (5.670400e-5) 
/** Ideal gas constant (\f$  \rm{erg}\, \rm{K}^{-1}\, \rm{mole}^{-1} \f$) */
#define RGAS            (8.3143e7) 
/** Electron volt in ergs */
#define EV              (1.60217653e-12) 
/** Thomson cross section in \f$ cm^2 \f$ */
#define SIGMA_THOMSON   (0.665245873e-24) 
/** Jansky (flux/frequency unit) in CGS */
#define JY              (1.e-23) 

/** Parsec */
#define PC              (3.085678e18) 
/** Astronomical Unit */
#define AU              (1.49597870691e13) 

/** Number of seconds in a year */
#define YEAR            (31536000.) 
/** Number of seconds in a day */
#define DAY             (86400.) 
/** Number of seconds in an hour */
#define HOUR            (3600.) 
/** Solar mass */
#define MSUN            (1.989e33) 
/** Radius of the Sun */
#define RSUN            (6.96e10) 
/** Luminosity of the Sun */
#define LSUN            (3.827e33) 
/** Temperature of the Sun's photosphere */
#define TSUN            (5.78e3) 
/** Earth's mass */
#define MEARTH          (5.976e27) 
/** Earth's radius */
#define REARTH          (6.378e8) 
/** Distance from Earth to Sgr A* */
#define DSGRA           (8.4e3 * PC) 

/** Cosmic Background Radiation (CBR) temperature (from COBE) */
#define TCBR            (2.726) 

/* Abundances (from M & B, p. 99) */

/** Hydrogen (H) abundance */
#define SOLX            (0.70) 
/** Helium (He) abundance */
#define SOLY            (0.28) 
/** Metals abundance */
#define SOLZ            (0.02) 
/** Ratio of \f$ 4/π \f$ */
#define FOUR_PI         (1.2732395447351626862) 
/** Ratio of \f$ 4/\pi^2 \f$ */
#define FOUR_PISQ       (0.40528473456935108578) 
/** Ratio of \f$ 3\pi/2 \f$ */
#define THREEPI_TWO     (4.7123889803846898577) 
/** Constant defined as \f$ 2^{11/12} \f$ */
#define CST             1.88774862536 

