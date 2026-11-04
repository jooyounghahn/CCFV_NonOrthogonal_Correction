/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
\*---------------------------------------------------------------------------*/

#include "ccfvNorms.H"

// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

Foam::ccfvCellErrors Foam::ccfvCellError
(
    const ccfvGeometry& geom,
    const scalarField& e
)
{
    const scalarField& measK = geom.measure();

    scalar sumE2 = 0;
    scalar maxE = 0;

    forAll(e, celli)
    {
        sumE2 += measK[celli]*e[celli]*e[celli];
        maxE = max(maxE, mag(e[celli]));
    }

    reduce(sumE2, sumOp<scalar>());
    reduce(maxE, maxOp<scalar>());

    ccfvCellErrors errors;
    errors.E2 = Foam::sqrt(sumE2);
    errors.Linf = maxE;

    return errors;
}


Foam::scalar Foam::ccfvDiscreteH1
(
    const ccfvGeometry& geom,
    const scalarField& e
)
{
    const fvMesh& mesh = geom.mesh();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const ccfvHalo& halo = geom.halo();

    scalar sum = 0;

    forAll(geom.internal(), facei)
    {
        const ccfvInteriorFacet& f = geom.internal()[facei];
        sum += f.magS*sqr(e[own[facei]] - e[nei[facei]])/(f.dK + f.dL);
    }

    // A processor face is seen by the ranks on both of its sides: each
    // rank adds one half
    const List<scalarField> eHalo(halo.exchange(e));

    for (const label patchi : halo.processorPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();

        forAll(faceCells, i)
        {
            const ccfvInteriorFacet& f = geom.processor()[patchi][i];
            sum +=
                0.5*f.magS*sqr(e[faceCells[i]] - eHalo[patchi][i])
               /(f.dK + f.dL);
        }
    }

    for (const label patchi : halo.dirichletPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();

        forAll(faceCells, i)
        {
            const ccfvBoundaryFacet& f = geom.boundary()[patchi][i];
            sum += f.magS*sqr(e[faceCells[i]])/f.dK;
        }
    }

    return Foam::sqrt(returnReduce(sum, sumOp<scalar>()));
}


// ************************************************************************* //
