/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
\*---------------------------------------------------------------------------*/

#include "ccfvFlux.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::ccfvFlux::ccfvFlux(const ccfvGeometry& geom, const ccfvScheme& scheme)
:
    geom_(geom),
    scheme_(scheme),
    aInt_(geom.internal().size()),
    aProc_(geom.mesh().boundary().size()),
    aBnd_(geom.mesh().boundary().size()),
    tInt_(geom.internal().size()),
    tProc_(geom.mesh().boundary().size()),
    tBnd_(geom.mesh().boundary().size()),
    diag_(geom.mesh().nCells(), Zero)
{
    const fvMesh& mesh = geom.mesh();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const ccfvHalo& halo = geom.halo();

    forAll(aInt_, facei)
    {
        const ccfvInteriorFacet& f = geom.internal()[facei];
        const scalar a = scheme.twoPointCoeff(f.S, f.d);

        aInt_[facei] = a;
        tInt_[facei] = f.S - a*f.d;

        diag_[own[facei]] += aInt_[facei];
        diag_[nei[facei]] += aInt_[facei];
    }

    for (const label patchi : halo.processorPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();
        aProc_[patchi].resize(faceCells.size());
        tProc_[patchi].resize(faceCells.size());

        forAll(faceCells, i)
        {
            const ccfvInteriorFacet& f = geom.processor()[patchi][i];
            const scalar a = scheme.twoPointCoeff(f.S, f.d);

            aProc_[patchi][i] = a;
            tProc_[patchi][i] = f.S - a*f.d;

            diag_[faceCells[i]] += aProc_[patchi][i];
        }
    }

    for (const label patchi : halo.dirichletPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();
        aBnd_[patchi].resize(faceCells.size());
        tBnd_[patchi].resize(faceCells.size());

        forAll(faceCells, i)
        {
            const ccfvBoundaryFacet& f = geom.boundary()[patchi][i];
            const scalar a = scheme.twoPointCoeff(f.S, f.d);

            aBnd_[patchi][i] = a;
            tBnd_[patchi][i] = f.S - a*f.d;

            diag_[faceCells[i]] += aBnd_[patchi][i];
        }
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::ccfvFlux::assemble
(
    lduMatrix& A,
    FieldField<Field, scalar>& interfaceBouCoeffs,
    FieldField<Field, scalar>& interfaceIntCoeffs
) const
{
    const fvMesh& mesh = geom_.mesh();

    A.diag() = diag_;
    A.upper() = -aInt_;

    // Zero coefficients on all patches, a_F on the processor patches
    const label nPatches = mesh.boundary().size();
    interfaceBouCoeffs.setSize(nPatches);
    interfaceIntCoeffs.setSize(nPatches);

    forAll(mesh.boundary(), patchi)
    {
        const label nFaces = mesh.boundary()[patchi].size();
        interfaceBouCoeffs.set(patchi, new scalarField(nFaces, Zero));
        interfaceIntCoeffs.set(patchi, new scalarField(nFaces, Zero));
    }

    for (const label patchi : geom_.halo().processorPatches())
    {
        interfaceBouCoeffs[patchi] = aProc_[patchi];
        interfaceIntCoeffs[patchi] = aProc_[patchi];
    }
}


Foam::tmp<Foam::scalarField> Foam::ccfvFlux::source
(
    const scalarField& rhs,
    const List<scalarField>& phiB,
    const vectorField& G,
    const List<vectorField>& GHalo
) const
{
    const fvMesh& mesh = geom_.mesh();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const ccfvHalo& halo = geom_.halo();

    auto tb = tmp<scalarField>::New(rhs);
    scalarField& b = tb.ref();

    // Internal faces: K = owner, L = neighbour
    forAll(aInt_, facei)
    {
        const ccfvInteriorFacet& f = geom_.internal()[facei];
        const label K = own[facei];
        const label L = nei[facei];

        const vector GF(scheme_.average(G[K], G[L], f.dK, f.dL));
        const scalar tG = tInt_[facei] & GF;

        b[K] += tG;
        b[L] -= tG;
    }

    // Processor faces: K = the cell on this rank, L = the halo cell
    for (const label patchi : halo.processorPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();

        forAll(faceCells, i)
        {
            const ccfvInteriorFacet& f = geom_.processor()[patchi][i];

            const vector GF
            (
                scheme_.average
                (
                    G[faceCells[i]],
                    GHalo[patchi][i],
                    f.dK,
                    f.dL
                )
            );

            b[faceCells[i]] += tProc_[patchi][i] & GF;
        }
    }

    // Boundary faces: {{G}}_F = G_K and the Dirichlet value phi(x_F)
    for (const label patchi : halo.dirichletPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();

        forAll(faceCells, i)
        {
            const label K = faceCells[i];

            b[K] +=
                aBnd_[patchi][i]*phiB[patchi][i]
              + (tBnd_[patchi][i] & G[K]);
        }
    }

    return tb;
}


Foam::tmp<Foam::scalarField> Foam::ccfvFlux::Amul
(
    const scalarField& psi,
    const List<scalarField>& psiHalo
) const
{
    const fvMesh& mesh = geom_.mesh();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();

    auto tApsi = tmp<scalarField>::New(diag_*psi);
    scalarField& Apsi = tApsi.ref();

    forAll(aInt_, facei)
    {
        Apsi[own[facei]] -= aInt_[facei]*psi[nei[facei]];
        Apsi[nei[facei]] -= aInt_[facei]*psi[own[facei]];
    }

    for (const label patchi : geom_.halo().processorPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();

        forAll(faceCells, i)
        {
            Apsi[faceCells[i]] -= aProc_[patchi][i]*psiHalo[patchi][i];
        }
    }

    return tApsi;
}


// ************************************************************************* //
