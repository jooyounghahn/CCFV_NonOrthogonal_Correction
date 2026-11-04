/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
-------------------------------------------------------------------------------
Application
    ccfvPoisson

Description
    Solves -Laplace(phi) = f in Omega, phi = phi_D on the boundary, with the
    cell-centered finite volume method of the paper "Stability and
    error analysis of cell-centered finite volume methods with
    non-orthogonal correction on polytopal meshes":
      - numerical flux with the minimum, full-area or over-relaxed
        correction,
      - weighted least-squares gradient with the weights omega_{K,F}
        (see ccfvScheme),
      - facet average with the weights alpha_{K,F} (see ccfvScheme).
    One source serves 2D meshes (one cell thick in z, empty front and back
    patches) and 3D meshes; see ccfvGeometry.

    Data as in the numerical tests of the paper: the right-hand side is the
    midpoint rule |K| f(x_K), and the boundary values are phi_h|_F = phi(x_F)
    at the facet centres x_F.

    The discrete problem A phi = b(phi) (see ccfvFlux) is solved by deferred
    correction: phi^0 = 0 and
        A phi^k = b(phi^{k-1}),   k = 1, 2, ...,
    where A is the two-point matrix and b contains |K| f(x_K), the boundary
    terms and the correction terms with the gradient of phi^{k-1}.  Each
    linear system is solved by the OpenFOAM solver of the entry "phi" in
    system/fvSolution.  The iteration stops at the first k with
        res = sum_K |(A phi^k - b(phi^k))_K| / sum_K |A_KK| < epsilon,
    i.e. when phi^k solves the discrete problem up to the relative residual epsilon.  It is
    reported as diverged if res is not finite or exceeds 1e8 times the
    residual of the first step (the factor can be changed with the optional
    entry divergeFactor).

Usage
    ccfvPoisson [-case dir] [-parallel]

    With -parallel the case must be decomposed first (decomposePar) and the
    program started with mpirun.  Reads system/ccfvDict:

        solution        example1;    // example1 | example2 | linear
        correction      overRelaxed; // minimum | fullArea | overRelaxed
        lsWeights       iii;         // i | ii | iii   (least-squares weights)
        averageWeights  iii;         // i | ii | iii   (facet average weights)
        epsilon         1e-12;       // stopping tolerance (default 1e-12)
        maxOuter        200;         // maximum number of steps (default 200)
        writeFields     false;       // write phi_h at time 1 (default false)

    The first four entries are required.  The convergence tables of the
    paper use overRelaxed, (iii) and (iii).

Output
    One line that starts with "RESULT ", with the entries
        dim nProcs nCells solution correction lsWeights averageWeights
        K res E2 H1h Linf linIter diverged time
    K       number of steps; negative (-K) if the stopping rule was not met
    res     final relative residual (see above)
    E2      ||phi*_I - phi_h||_{0,Omega}  (phi*_I|_K = phi(x_K))
    H1h     |phi*_I - phi_h|_h            (discrete H1 seminorm)
    Linf    max_K |(phi_h)_K - phi(x_K)|
    linIter total number of iterations of the linear solver
    diverged 1 if the iteration diverged, else 0
    time    wall-clock time in seconds after reading the mesh
    Floating-point values are written as %.10e.

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "calculatedFvPatchFields.H"
#include "IOdictionary.H"
#include "lduMatrix.H"
#include "MinMax.H"

#include "ccfvFormat.H"
#include "ccfvHalo.H"
#include "ccfvGeometry.H"
#include "ccfvScheme.H"
#include "ccfvGradient.H"
#include "ccfvFlux.H"
#include "ccfvNorms.H"
#include "ccfvExactSolution.H"

#include <chrono>
#include <cmath>

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Cell-centered finite volume method with non-orthogonal correction"
        " for -Laplace(phi) = f with Dirichlet data; settings in"
        " system/ccfvDict"
    );
    argList::noFunctionObjects();

    #include "setRootCase.H"

    // No function objects and no libraries of the controlDict entry 'libs'
    Time runTime(Time::controlDictName, args, false, false);

    fvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.timeName(),
            runTime,
            IOobject::MUST_READ
        )
    );

    const IOdictionary ccfvDict
    (
        IOobject
        (
            "ccfvDict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const ccfvExactSolution::problemType problem =
        ccfvExactSolution::problemNames.get("solution", ccfvDict);
    const ccfvScheme scheme(ccfvDict);
    const scalar epsilon = ccfvDict.getOrDefault<scalar>("epsilon", 1e-12);
    const label maxOuter =
        ccfvDict.getCheckOrDefault<label>("maxOuter", 200, labelMinMax::ge(1));
    const bool writeFields = ccfvDict.getOrDefault("writeFields", false);
    const scalar divergeFactor =
        ccfvDict.getOrDefault<scalar>("divergeFactor", 1e8);

    Info<< nl << "ccfvPoisson" << nl
        << "    solution       " << ccfvExactSolution::problemNames.get(problem)
        << nl
        << "    correction     " << scheme.correctionName() << "  (non-orthogonal correction)" << nl
        << "    lsWeights      " << scheme.lsWeightsName()
        << "  (least-squares weights)" << nl
        << "    averageWeights " << scheme.averageWeightsName()
        << "  (facet average weights)" << nl
        << "    epsilon        " << ccfvSci(epsilon) << nl
        << "    maxOuter       " << maxOuter << nl
        << "    nProcs         " << UPstream::nProcs() << nl << endl;

    const auto clock0 = std::chrono::steady_clock::now();

    // Mesh quantities
    const ccfvHalo halo(mesh);
    const ccfvGeometry geom(mesh, halo);

    if (geom.twoD() && geom.emptyDirection() != vector::Z)
    {
        FatalErrorInFunction
            << "The 2D problems are posed in the x-y plane, but the empty"
            << " direction of the mesh is " << label(geom.emptyDirection())
            << " (z is 2)" << exit(FatalError);
    }

    const ccfvExactSolution exact(problem, geom.dim());

    const label nCells = returnReduce(mesh.nCells(), sumOp<label>());

    Info<< "Mesh: dimension d = " << geom.dim() << ", " << nCells << " cells";
    if (geom.twoD())
    {
        Info<< " (2D slab of thickness " << ccfvSci(geom.thickness())
            << "; |K| = area, |F| = edge length)";
    }
    Info<< nl << endl;

    const ccfvGradient lsGradient(geom, scheme);

    Info<< "Least-squares matrices: min_K det(M_K)/(tr(M_K)/3)^3 = "
        << ccfvSci(lsGradient.minDetRatio()) << nl << endl;

    // Right-hand side |K| f(x_K) and boundary values phi(x_F)
    const vectorField& xK = mesh.C().primitiveField();
    const scalarField& measK = geom.measure();

    scalarField rhs(mesh.nCells());
    forAll(xK, celli)
    {
        rhs[celli] = measK[celli]*exact.f(xK[celli]);
    }

    List<scalarField> phiB(mesh.boundary().size());
    for (const label patchi : halo.dirichletPatches())
    {
        const vectorField& xF = mesh.Cf().boundaryField()[patchi];
        phiB[patchi].resize(xF.size());
        forAll(xF, i)
        {
            phiB[patchi][i] = exact.phi(xF[i]);
        }
    }

    // The unknown; its processor patch fields provide the interfaces of
    // the parallel linear solver
    volScalarField phiField
    (
        IOobject
        (
            "phi",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimless, Zero),
        calculatedFvPatchScalarField::typeName
    );
    scalarField& phi = phiField.primitiveFieldRef();

    // Two-point matrix A and its linear solver
    const ccfvFlux flux(geom, scheme);

    lduMatrix A(mesh);
    FieldField<Field, scalar> interfaceBouCoeffs;
    FieldField<Field, scalar> interfaceIntCoeffs;
    flux.assemble(A, interfaceBouCoeffs, interfaceIntCoeffs);

    const lduInterfaceFieldPtrsList interfaces
    (
        phiField.boundaryField().scalarInterfaces()
    );

    autoPtr<lduMatrix::solver> linearSolver = lduMatrix::solver::New
    (
        "phi",
        A,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        mesh.solverDict("phi")
    );

    const scalar sumDiag =
        returnReduce(sum(mag(flux.diag())), sumOp<scalar>());

    // Deferred correction, starting from phi^0 = 0
    phi = Zero;
    List<scalarField> phiHalo(halo.exchange(phi));
    vectorField G(lsGradient.grad(phi, phiHalo, phiB));
    scalarField b(flux.source(rhs, phiB, G, halo.exchange(G)));

    label K = 0;
    label linIter = 0;
    scalar res = 0;
    scalar res1 = 0;
    bool converged = false;
    bool diverged = false;

    for (label k = 1; k <= maxOuter; ++k)
    {
        const solverPerformance perf = linearSolver->solve(phi, b);
        linIter += perf.nIterations();
        K = k;

        phiHalo = halo.exchange(phi);
        G = lsGradient.grad(phi, phiHalo, phiB);
        b = flux.source(rhs, phiB, G, halo.exchange(G));

        res =
            returnReduce
            (
                sum(mag(flux.Amul(phi, phiHalo) - b)),
                sumOp<scalar>()
            )/sumDiag;

        if (k == 1)
        {
            res1 = res;
        }

        Info<< "iter " << k << "  res " << ccfvSci(res)
            << "  linIters " << perf.nIterations() << nl
            << "    " << perf.solverName()
            << ": initial " << ccfvSci(perf.initialResidual())
            << ", final " << ccfvSci(perf.finalResidual())
            << (perf.converged() ? "" : "  (linear solver not converged)")
            << endl;

        if (res < epsilon)
        {
            converged = true;
            break;
        }
        if (!std::isfinite(res) || res > divergeFactor*res1)
        {
            diverged = true;
            WarningInFunction
                << "The iteration diverged at step " << k << endl;
            break;
        }
    }
    Info<< endl;

    if (!converged)
    {
        WarningInFunction
            << "Stopping rule not met: res = " << ccfvSci(res)
            << ", RESULT reports K = " << -K << nl << endl;
        K = -K;
    }

    // Errors against the pointwise interpolant phi*_I|_K = phi(x_K)
    scalarField phiStar(mesh.nCells());
    forAll(xK, celli)
    {
        phiStar[celli] = exact.phi(xK[celli]);
    }
    const scalarField e(phi - phiStar);

    const ccfvCellErrors cellErrors = ccfvCellError(geom, e);
    const scalar H1h = ccfvDiscreteH1(geom, e);

    const scalar elapsed =
        returnReduce
        (
            std::chrono::duration<scalar>
            (
                std::chrono::steady_clock::now() - clock0
            ).count(),
            maxOp<scalar>()
        );

    Info<< "RESULT"
        << " dim=" << geom.dim()
        << " nProcs=" << UPstream::nProcs()
        << " nCells=" << nCells
        << " solution=" << exact.name()
        << " correction=" << scheme.correctionName()
        << " lsWeights=" << scheme.lsWeightsName()
        << " averageWeights=" << scheme.averageWeightsName()
        << " K=" << K
        << " res=" << ccfvSci(res)
        << " E2=" << ccfvSci(cellErrors.E2)
        << " H1h=" << ccfvSci(H1h)
        << " Linf=" << ccfvSci(cellErrors.Linf)
        << " linIter=" << linIter
        << " diverged=" << label(diverged ? 1 : 0)
        << " time=" << ccfvSci(elapsed)
        << endl;

    if (writeFields)
    {
        ++runTime;

        // Boundary values phi(x_F); on processor patches the halo values
        for (const label patchi : halo.dirichletPatches())
        {
            phiField.boundaryFieldRef()[patchi] == phiB[patchi];
        }
        for (const label patchi : halo.processorPatches())
        {
            phiField.boundaryFieldRef()[patchi] == phiHalo[patchi];
        }

        phiField.write();

        Info<< nl << "Wrote phi at time " << runTime.timeName() << endl;
    }

    Info<< nl << "End" << nl << endl;

    return 0;
}


// ************************************************************************* //
