/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
-------------------------------------------------------------------------------
Class
    Foam::ccfvGradient

Description
    Weighted least-squares gradient G_h of the paper:

        (G_h psi)|_K = argmin_v sum_{F in F_K} omega_{K,F}
                              |v.r_{K,F} - grad_r^{K,F} psi|^2,

    i.e. M_K (G_h psi)|_K = c_K with
        M_K = sum_F omega_{K,F} r_{K,F} r_{K,F}^T,
        c_K = sum_F omega_{K,F} (grad_r^{K,F} psi) r_{K,F},
    where grad_r^{K,F} psi = (psi_L - psi_K)/delta_F on an interior facet
    and (psi_F - psi_K)/delta_F on a boundary facet; psi_F = phi(x_F) is the
    Dirichlet value.  The weights omega_{K,F} are those of ccfvScheme
    (ccfvScheme::omega).

    With the vector d = delta_F r_{K,F} (d = x_L - x_K, or x_F - x_K) and the
    weight w = omega_{K,F}/delta_F^2 this reads
        M_K = sum_F w d d^T,   c_K = sum_F w (psi_L - psi_K) d.
    M_K is inverted once per cell (cofactor formula), and for every facet
    side the vector M_K^{-1} w d is stored, so that the gradient is a sum
    of these vectors times the differences psi_L - psi_K.

    2D: all vectors d have a zero z component, so M_K has a zero row and
    column in that direction.  Adding tr(M_K) to the zz entry makes M_K
    invertible without changing the in-plane gradient, whose z component is
    then zero.

    The constructor stops with an error if a matrix is close to singular:
    det(M_K)/(tr(M_K)/3)^3 < 1e-12.

SourceFiles
    ccfvGradient.C

\*---------------------------------------------------------------------------*/

#ifndef ccfvGradient_H
#define ccfvGradient_H

#include "ccfvGeometry.H"
#include "ccfvScheme.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class ccfvGradient Declaration
\*---------------------------------------------------------------------------*/

class ccfvGradient
{
    // Private Data

        const ccfvGeometry& geom_;

        //- Internal faces: M^{-1} w d of the owner and minus that of the
        //- neighbour
        vectorField coeffOwn_;
        vectorField coeffNei_;

        //- Processor faces [patchi][facei]: M^{-1} w d of the local cell
        List<vectorField> coeffProc_;

        //- Boundary faces [patchi][facei]: M^{-1} w d of the cell
        List<vectorField> coeffBnd_;

        //- Global minimum of det(M_K)/(tr(M_K)/3)^3
        scalar minDetRatio_;


public:

    // Constructors

        //- Build and invert the matrices M_K for the weights of the scheme
        ccfvGradient(const ccfvGeometry& geom, const ccfvScheme& scheme);


    // Member Functions

        //- Global minimum of det(M_K)/(tr(M_K)/3)^3
        scalar minDetRatio() const noexcept { return minDetRatio_; }

        //- The gradient G_h psi from the cell values psi, the halo values
        //- psiHalo [patchi][facei] (ccfvHalo::exchange) and the boundary
        //- values psiB [patchi][facei]
        tmp<vectorField> grad
        (
            const scalarField& psi,
            const List<scalarField>& psiHalo,
            const List<scalarField>& psiB
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
