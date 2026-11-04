/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
\*---------------------------------------------------------------------------*/

#include "ccfvGradient.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::ccfvGradient::ccfvGradient
(
    const ccfvGeometry& geom,
    const ccfvScheme& scheme
)
:
    geom_(geom),
    coeffOwn_(geom.internal().size()),
    coeffNei_(geom.internal().size()),
    coeffProc_(geom.mesh().boundary().size()),
    coeffBnd_(geom.mesh().boundary().size()),
    minDetRatio_(GREAT)
{
    const fvMesh& mesh = geom.mesh();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const ccfvHalo& halo = geom.halo();
    const scalarField& measK = geom.measure();

    // w = omega_{K,F}/delta_F^2 for one side of an interior facet, given
    // alpha_{K,F} d_F of that side and the measure of its cell
    auto interiorWeight =
        [&scheme](const ccfvInteriorFacet& f, scalar aDF, scalar meas)
        {
            const scalar deltaF = mag(f.d);
            return
                scheme.omega(aDF, f.dK + f.dL, deltaF, f.magS, meas)
               /sqr(deltaF);
        };

    // w for a boundary facet: alpha_{K,F} = 1 and d_F = d_{K,F}
    auto boundaryWeight =
        [&scheme](const ccfvBoundaryFacet& f, scalar meas)
        {
            const scalar deltaF = mag(f.d);
            return scheme.omega(f.dK, f.dK, deltaF, f.magS, meas)/sqr(deltaF);
        };

    scalarField wOwn(geom.internal().size());
    scalarField wNei(geom.internal().size());
    List<scalarField> wProc(mesh.boundary().size());
    List<scalarField> wBnd(mesh.boundary().size());

    // M_K = sum_F w d d^T  (= sum_F omega r r^T)
    symmTensorField M(mesh.nCells(), Zero);

    forAll(geom.internal(), facei)
    {
        const ccfvInteriorFacet& f = geom.internal()[facei];
        const symmTensor dd(sqr(f.d));

        wOwn[facei] =
            interiorWeight(f, scheme.alphaDF(f.dK, f.dL), measK[own[facei]]);
        wNei[facei] =
            interiorWeight(f, scheme.alphaDF(f.dL, f.dK), measK[nei[facei]]);

        M[own[facei]] += wOwn[facei]*dd;
        M[nei[facei]] += wNei[facei]*dd;
    }

    for (const label patchi : halo.processorPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();
        wProc[patchi].resize(faceCells.size());

        forAll(faceCells, i)
        {
            const ccfvInteriorFacet& f = geom.processor()[patchi][i];

            wProc[patchi][i] =
                interiorWeight
                (
                    f,
                    scheme.alphaDF(f.dK, f.dL),
                    measK[faceCells[i]]
                );

            M[faceCells[i]] += wProc[patchi][i]*sqr(f.d);
        }
    }

    for (const label patchi : halo.dirichletPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();
        wBnd[patchi].resize(faceCells.size());

        forAll(faceCells, i)
        {
            const ccfvBoundaryFacet& f = geom.boundary()[patchi][i];

            wBnd[patchi][i] = boundaryWeight(f, measK[faceCells[i]]);

            M[faceCells[i]] += wBnd[patchi][i]*sqr(f.d);
        }
    }

    // 2D: tr(M_K) in the empty direction makes M_K invertible
    if (geom.twoD())
    {
        static const direction diagonal[3] =
            {symmTensor::XX, symmTensor::YY, symmTensor::ZZ};

        const direction kk = diagonal[geom.emptyDirection()];

        forAll(M, celli)
        {
            M[celli].component(kk) += tr(M[celli]);
        }
    }

    // M_K^{-1} = adj(M_K)/det(M_K) of the symmetric 3x3 matrices
    symmTensorField Minv(M.size());

    forAll(M, celli)
    {
        const symmTensor& m = M[celli];

        const scalar adjXX = m.yy()*m.zz() - m.yz()*m.yz();
        const scalar adjXY = m.xz()*m.yz() - m.xy()*m.zz();
        const scalar adjXZ = m.xy()*m.yz() - m.xz()*m.yy();
        const scalar det = m.xx()*adjXX + m.xy()*adjXY + m.xz()*adjXZ;

        const scalar meanDiag = (m.xx() + m.yy() + m.zz())/3.0;
        const scalar ratio = det/(meanDiag*meanDiag*meanDiag);
        minDetRatio_ = min(minDetRatio_, ratio);

        if (ratio < 1e-12)
        {
            FatalErrorInFunction
                << "The least-squares matrix of cell " << celli
                << " is close to singular: det(M)/(tr(M)/3)^3 = " << ratio
                << exit(FatalError);
        }

        Minv[celli] =
            symmTensor
            (
                adjXX, adjXY, adjXZ,
                m.xx()*m.zz() - m.xz()*m.xz(),
                m.xy()*m.xz() - m.xx()*m.yz(),
                m.xx()*m.yy() - m.xy()*m.xy()
            )/det;
    }
    reduce(minDetRatio_, minOp<scalar>());

    // Coefficient vectors M_K^{-1} w d of every facet side
    forAll(geom.internal(), facei)
    {
        const vector& d = geom.internal()[facei].d;

        // For the neighbour L the vector is x_K - x_L = -d and the
        // difference is psi_K - psi_L: stored with a minus sign and
        // subtracted in grad()
        coeffOwn_[facei] = Minv[own[facei]] & (wOwn[facei]*d);
        coeffNei_[facei] = -(Minv[nei[facei]] & (wNei[facei]*d));
    }

    for (const label patchi : halo.processorPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();
        vectorField& coeff = coeffProc_[patchi];
        coeff.resize(faceCells.size());

        forAll(faceCells, i)
        {
            coeff[i] =
                Minv[faceCells[i]]
              & (wProc[patchi][i]*geom.processor()[patchi][i].d);
        }
    }

    for (const label patchi : halo.dirichletPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();
        vectorField& coeff = coeffBnd_[patchi];
        coeff.resize(faceCells.size());

        forAll(faceCells, i)
        {
            coeff[i] =
                Minv[faceCells[i]]
              & (wBnd[patchi][i]*geom.boundary()[patchi][i].d);
        }
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::vectorField> Foam::ccfvGradient::grad
(
    const scalarField& psi,
    const List<scalarField>& psiHalo,
    const List<scalarField>& psiB
) const
{
    const fvMesh& mesh = geom_.mesh();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const ccfvHalo& halo = geom_.halo();

    auto tG = tmp<vectorField>::New(mesh.nCells(), Zero);
    vectorField& G = tG.ref();

    forAll(coeffOwn_, facei)
    {
        const label K = own[facei];
        const label L = nei[facei];
        const scalar jump = psi[L] - psi[K];

        G[K] += coeffOwn_[facei]*jump;
        G[L] -= coeffNei_[facei]*jump;
    }

    for (const label patchi : halo.processorPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();
        const vectorField& coeff = coeffProc_[patchi];
        const scalarField& psiL = psiHalo[patchi];

        forAll(faceCells, i)
        {
            G[faceCells[i]] += coeff[i]*(psiL[i] - psi[faceCells[i]]);
        }
    }

    for (const label patchi : halo.dirichletPatches())
    {
        const labelUList& faceCells = mesh.boundary()[patchi].faceCells();
        const vectorField& coeff = coeffBnd_[patchi];
        const scalarField& psiF = psiB[patchi];

        forAll(faceCells, i)
        {
            G[faceCells[i]] += coeff[i]*(psiF[i] - psi[faceCells[i]]);
        }
    }

    return tG;
}


// ************************************************************************* //
