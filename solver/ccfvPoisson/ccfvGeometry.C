/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
\*---------------------------------------------------------------------------*/

#include "ccfvGeometry.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

Foam::ccfvInteriorFacet Foam::ccfvGeometry::interiorFacet
(
    const vector& S,
    const vector& xF,
    const vector& xK,
    const vector& xL
)
{
    ccfvInteriorFacet f;

    f.S = S;
    f.magS = mag(S);

    // Unit normal n_{K,F} and the distances of x_K, x_L to the facet plane
    const vector n(S/f.magS);
    const vector xKF(xF - xK);
    const vector xLF(xF - xL);
    const scalar nKF = n & xKF;
    const scalar nLF = n & xLF;

    f.d = xL - xK;
    f.dK = mag(nKF);
    f.dL = mag(nLF);

    return f;
}


Foam::ccfvBoundaryFacet Foam::ccfvGeometry::boundaryFacet
(
    const vector& S,
    const vector& xF,
    const vector& xK
)
{
    ccfvBoundaryFacet f;

    f.S = S;
    f.magS = mag(S);

    const vector n(S/f.magS);
    f.d = xF - xK;
    const scalar nKF = n & f.d;

    f.dK = mag(nKF);

    return f;
}


void Foam::ccfvGeometry::setDimension()
{
    const bool anyEmpty = returnReduceOr(!halo_.emptyPatches().empty());

    if (!anyEmpty)
    {
        dim_ = 3;
        emptyDir_ = 0;
        thickness_ = 1;
        measure_ = mesh_.V().field();
        return;
    }

    // Mesh with empty patches: the 2D problem in the plane normal to the
    // single empty direction
    dim_ = 2;

    if (mesh_.nGeometricD() != 2)
    {
        FatalErrorInFunction
            << "The mesh has empty patches but " << mesh_.nGeometricD()
            << " geometric directions (a 2D mesh needs 2)"
            << exit(FatalError);
    }

    const Vector<label>& geomDirs = mesh_.geometricD();
    label k = -1;
    for (direction cmpt = 0; cmpt < vector::nComponents; ++cmpt)
    {
        if (geomDirs[cmpt] < 0)
        {
            k = cmpt;
        }
    }
    if (k < 0)
    {
        FatalErrorInFunction
            << "No empty direction found in a mesh with empty patches"
            << exit(FatalError);
    }
    emptyDir_ = direction(k);

    // Every point must lie on one of the two bounding planes of the slab
    const scalarField zeta(mesh_.points().component(emptyDir_));
    const scalar zetaMin = returnReduce(min(zeta), minOp<scalar>());
    const scalar zetaMax = returnReduce(max(zeta), maxOp<scalar>());
    thickness_ = zetaMax - zetaMin;

    label nOffPlane = 0;
    forAll(zeta, pointi)
    {
        if
        (
            mag(zeta[pointi] - zetaMin) > 1e-10*thickness_
         && mag(zeta[pointi] - zetaMax) > 1e-10*thickness_
        )
        {
            ++nOffPlane;
        }
    }
    reduce(nOffPlane, sumOp<label>());

    if (thickness_ <= 0 || nOffPlane)
    {
        FatalErrorInFunction
            << "The 2D mesh is not a slab of one cell: thickness "
            << thickness_ << ", points between the two planes: " << nOffPlane
            << exit(FatalError);
    }

    // |K| = cell area
    measure_ = mesh_.V().field()/thickness_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::ccfvGeometry::ccfvGeometry(const fvMesh& mesh, const ccfvHalo& halo)
:
    mesh_(mesh),
    halo_(halo),
    internal_(mesh.nInternalFaces()),
    processor_(mesh.boundary().size()),
    boundary_(mesh.boundary().size()),
    dim_(3),
    emptyDir_(0),
    thickness_(1),
    measure_()
{
    setDimension();

    // 2D: points and area vectors in the plane, area vectors divided by t
    const scalar invThickness = 1/thickness_;
    auto area = [&](const vector& v)
    {
        return twoD() ? invThickness*inPlane(v) : v;
    };
    auto point = [&](const vector& v)
    {
        return twoD() ? inPlane(v) : v;
    };

    const vectorField& C = mesh.C().primitiveField();
    const vectorField& Cf = mesh.Cf().primitiveField();
    const vectorField& Sf = mesh.Sf().primitiveField();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();

    forAll(internal_, facei)
    {
        internal_[facei] =
            interiorFacet
            (
                area(Sf[facei]),
                point(Cf[facei]),
                point(C[own[facei]]),
                point(C[nei[facei]])
            );
    }

    // Centres of the cells on the other side of the processor faces
    const List<vectorField> haloCentres(halo.exchange(C));

    for (const label patchi : halo.processorPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();
        const vectorField& pCf = mesh.Cf().boundaryField()[patchi];
        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
        const vectorField& xL = haloCentres[patchi];

        List<ccfvInteriorFacet>& facets = processor_[patchi];
        facets.resize(faceCells.size());

        forAll(faceCells, i)
        {
            facets[i] =
                interiorFacet
                (
                    area(pSf[i]),
                    point(pCf[i]),
                    point(C[faceCells[i]]),
                    point(xL[i])
                );
        }
    }

    for (const label patchi : halo.dirichletPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();
        const vectorField& pCf = mesh.Cf().boundaryField()[patchi];
        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];

        List<ccfvBoundaryFacet>& facets = boundary_[patchi];
        facets.resize(faceCells.size());

        forAll(faceCells, i)
        {
            facets[i] =
                boundaryFacet
                (
                    area(pSf[i]),
                    point(pCf[i]),
                    point(C[faceCells[i]])
                );
        }
    }
}


// ************************************************************************* //
