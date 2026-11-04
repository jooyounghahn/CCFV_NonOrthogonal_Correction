/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
-------------------------------------------------------------------------------
Description
    Error norms of the convergence tables of the paper for the cell error
    e_K = (phi_h)_K - phi(x_K), where phi*_I|_K = phi(x_K) is the pointwise
    interpolant.  All values are reduced over the ranks.

        E2   = ||phi*_I - phi_h||_{0,Omega} = ( sum_K |K| e_K^2 )^{1/2}
        Linf = max_K |e_K|
        H1h  = |phi*_I - phi_h|_h, the discrete H1 seminorm,
               |e|_h^2 = sum_{F interior} |F| (e_K - e_L)^2 / d_F
                       + sum_{F boundary} |F| e_K^2 / d_F,
               with d_F = d_{K,F} + d_{L,F} or d_F = d_{K,F}.
    On a boundary facet the error is zero, since phi_h|_F and phi*_I|_F
    both equal phi(x_F).

SourceFiles
    ccfvNorms.C

\*---------------------------------------------------------------------------*/

#ifndef ccfvNorms_H
#define ccfvNorms_H

#include "ccfvGeometry.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

//- The two cell norms of the error
struct ccfvCellErrors
{
    scalar E2;      //!< ( sum_K |K| e_K^2 )^{1/2}
    scalar Linf;    //!< max_K |e_K|
};


//- E2 and Linf of the cell error e
ccfvCellErrors ccfvCellError(const ccfvGeometry& geom, const scalarField& e);


//- The discrete H1 seminorm |e|_h of a cell function e that is zero on the
//- boundary facets (a norm for such functions)
scalar ccfvDiscreteH1(const ccfvGeometry& geom, const scalarField& e);


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
