/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
-------------------------------------------------------------------------------
Class
    Foam::ccfvFlux

Description
    The discrete problem of the paper with the numerical flux,

        J_{K,F}(psi) = -( c_F grad_r^{K,F} psi + {{G_h psi}}_F . e_{K,F} ),
        e_{K,F} = n_{K,F} - c_F r_{K,F},

    split into a two-point matrix and an explicit part for the deferred
    correction.  With S = |F| n_{K,F}, d = x_L - x_K (x_F - x_K on a
    boundary facet), a_F = |F| c_F/delta_F and t = |F| e_{K,F} = S - a_F d
    (ccfvScheme::twoPointCoeff), the integral of the flux over F is

        int_F J_{K,F}(psi) = a_F (psi_K - psi_L) - t.{{G_h psi}}_F,

    with psi_L = phi(x_F) on a boundary facet, where {{G_h psi}}_F = G_K.
    Hence the discrete problem with the right-hand side |K| f(x_K) reads A phi = b(phi):

        (A phi)_K = sum_F a_F phi_K - sum_{F interior} a_F phi_L,
        b(psi)_K  = |K| f(x_K) + sum_{F boundary} a_F phi(x_F)
                  + sum_F t.{{G_h psi}}_F.

    The facet average {{.}}_F uses the weights alpha_{K,F}
    (ccfvScheme::average).  Since t changes sign with the orientation of F,
    the contribution of an internal face is added to its owner and
    subtracted from its neighbour.

    lduMatrix storage: diag = A_KK, upper = -a_F of the internal faces; for
    the processor faces interfaceBouCoeffs = interfaceIntCoeffs = a_F.

SourceFiles
    ccfvFlux.C

\*---------------------------------------------------------------------------*/

#ifndef ccfvFlux_H
#define ccfvFlux_H

#include "ccfvGeometry.H"
#include "ccfvScheme.H"
#include "lduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                          Class ccfvFlux Declaration
\*---------------------------------------------------------------------------*/

class ccfvFlux
{
    // Private Data

        const ccfvGeometry& geom_;

        const ccfvScheme& scheme_;

        //- a_F: internal, processor [patchi][facei], boundary [patchi][facei]
        scalarField aInt_;
        List<scalarField> aProc_;
        List<scalarField> aBnd_;

        //- t = |F| e_{K,F}: internal, processor, boundary
        vectorField tInt_;
        List<vectorField> tProc_;
        List<vectorField> tBnd_;

        //- A_KK = sum_F a_F
        scalarField diag_;


public:

    // Constructors

        //- Compute a_F and t of every facet for the correction of the scheme
        ccfvFlux(const ccfvGeometry& geom, const ccfvScheme& scheme);


    // Member Functions

        //- The diagonal A_KK
        const scalarField& diag() const noexcept { return diag_; }

        //- Fill the lduMatrix and its processor interface coefficients
        void assemble
        (
            lduMatrix& A,
            FieldField<Field, scalar>& interfaceBouCoeffs,
            FieldField<Field, scalar>& interfaceIntCoeffs
        ) const;

        //- b(psi) from rhs = |K| f(x_K), the boundary values phiB
        //- [patchi][facei], the gradient G = G_h psi and its halo values
        //- GHalo [patchi][facei]
        tmp<scalarField> source
        (
            const scalarField& rhs,
            const List<scalarField>& phiB,
            const vectorField& G,
            const List<vectorField>& GHalo
        ) const;

        //- A psi with the halo values psiHalo [patchi][facei]
        tmp<scalarField> Amul
        (
            const scalarField& psi,
            const List<scalarField>& psiHalo
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
