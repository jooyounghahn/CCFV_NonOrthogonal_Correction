/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
-------------------------------------------------------------------------------
Class
    Foam::ccfvGeometry

Description
    Facet and cell quantities of the paper, computed once.
    As in the numerical tests of the paper, they are those of OpenFOAM: the cell centre
    x_K = mesh.C(), the facet centre x_F = mesh.Cf(), the area vector
    |F| n = mesh.Sf() and the cell volume mesh.V().

    Interior facet F with the cells K and L (an internal face, K = owner,
    or a face of a processor patch, K = the cell on this rank):
        S  = |F| n_{K,F}                   area vector, from K towards L
        d  = x_L - x_K                     |d| = delta_F, d/|d| = r_{K,F}
        dK = d_{K,F} = |n.(x_F - x_K)|     distance of x_K to the plane of F
        dL = d_{L,F} = |n.(x_F - x_L)|
    with n = S/|S|, so that d_F = dK + dL.  On a processor patch, x_L is the
    centre of the cell on the neighbouring rank (ccfvHalo::exchange).

    Boundary facet F of the cell K:
        S  = |F| n_{K,F},   d = x_F - x_K (|d| = delta_F),
        dK = d_{K,F} = |n.d| = d_F.

    Two-dimensional problem: a mesh with empty patches is a slab one cell
    thick in its empty direction (z for the tests; ccfvPoisson requires z),
    of thickness t.  The problem is then posed in the plane normal to that
    direction (dimension d = 2).  The empty-direction components of all
    points and area vectors are set to zero, the area vector is divided by t
    (so that |F| is the length of the edge) and |K| = V/t is the area of the
    cell.  For a mesh without empty patches (d = 3) the OpenFOAM values are
    used as they are and |K| = V.

SourceFiles
    ccfvGeometry.C

\*---------------------------------------------------------------------------*/

#ifndef ccfvGeometry_H
#define ccfvGeometry_H

#include "ccfvHalo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

//- Quantities of an interior facet (internal face or processor face)
struct ccfvInteriorFacet
{
    scalar magS;    //!< |F|
    vector S;       //!< |F| n_{K,F}
    vector d;       //!< x_L - x_K
    scalar dK;      //!< d_{K,F}
    scalar dL;      //!< d_{L,F}
};


//- Quantities of a boundary facet
struct ccfvBoundaryFacet
{
    scalar magS;    //!< |F|
    vector S;       //!< |F| n_{K,F}
    vector d;       //!< x_F - x_K
    scalar dK;      //!< d_{K,F} = d_F
};


/*---------------------------------------------------------------------------*\
                        Class ccfvGeometry Declaration
\*---------------------------------------------------------------------------*/

class ccfvGeometry
{
    // Private Data

        //- The mesh
        const fvMesh& mesh_;

        //- Patch sorting and halo exchange
        const ccfvHalo& halo_;

        //- Internal faces, indexed like the OpenFOAM internal faces
        List<ccfvInteriorFacet> internal_;

        //- Processor faces [patchi][facei]
        List<List<ccfvInteriorFacet>> processor_;

        //- Boundary faces of the Dirichlet patches [patchi][facei]
        List<List<ccfvBoundaryFacet>> boundary_;

        //- Dimension d of the problem: 3, or 2 for a mesh with empty patches
        label dim_;

        //- Empty direction of a 2D mesh
        direction emptyDir_;

        //- Thickness of a 2D mesh (1 for a 3D mesh)
        scalar thickness_;

        //- Cell measures |K|: volumes (d = 3) or areas (d = 2)
        scalarField measure_;


    // Private Member Functions

        //- Set dim_, emptyDir_, thickness_ and measure_
        void setDimension();

        //- The vector with its empty-direction component set to zero (2D)
        vector inPlane(const vector& v) const
        {
            vector p(v);
            if (dim_ == 2)
            {
                p.component(emptyDir_) = 0;
            }
            return p;
        }

        //- Quantities of an interior facet from its area vector S, its
        //- centre xF and the centres xK, xL of its two cells
        static ccfvInteriorFacet interiorFacet
        (
            const vector& S,
            const vector& xF,
            const vector& xK,
            const vector& xL
        );

        //- Quantities of a boundary facet from its area vector S, its
        //- centre xF and the centre xK of its cell
        static ccfvBoundaryFacet boundaryFacet
        (
            const vector& S,
            const vector& xF,
            const vector& xK
        );


public:

    // Constructors

        //- Compute all quantities; exchanges the halo cell centres once
        ccfvGeometry(const fvMesh& mesh, const ccfvHalo& halo);


    // Member Functions

        //- The mesh
        const fvMesh& mesh() const noexcept { return mesh_; }

        //- Patch sorting and halo exchange
        const ccfvHalo& halo() const noexcept { return halo_; }

        //- Internal faces
        const List<ccfvInteriorFacet>& internal() const noexcept
        {
            return internal_;
        }

        //- Processor faces [patchi][facei]
        const List<List<ccfvInteriorFacet>>& processor() const noexcept
        {
            return processor_;
        }

        //- Boundary faces of the Dirichlet patches [patchi][facei]
        const List<List<ccfvBoundaryFacet>>& boundary() const noexcept
        {
            return boundary_;
        }

        //- Dimension d of the problem (2 or 3)
        label dim() const noexcept { return dim_; }

        //- True for a 2D mesh
        bool twoD() const noexcept { return dim_ == 2; }

        //- Empty direction of a 2D mesh
        direction emptyDirection() const noexcept { return emptyDir_; }

        //- Thickness of a 2D mesh (1 for a 3D mesh)
        scalar thickness() const noexcept { return thickness_; }

        //- Cell measures |K|
        const scalarField& measure() const noexcept { return measure_; }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
