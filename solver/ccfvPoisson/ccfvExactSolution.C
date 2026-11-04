/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
\*---------------------------------------------------------------------------*/

#include "ccfvExactSolution.H"
#include "error.H"

// * * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * //

const Foam::Enum<Foam::ccfvExactSolution::problemType>
Foam::ccfvExactSolution::problemNames
({
    { problemType::example1, "example1" },
    { problemType::example2, "example2" },
    { problemType::linear, "linear" },
});


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::ccfvExactSolution::ccfvExactSolution
(
    const problemType problem,
    const label dim
)
:
    problem_(problem),
    dim_(dim)
{
    if (problem_ == problemType::example1 && dim_ != 2)
    {
        FatalErrorInFunction
            << "solution example1 is the 2D problem of Example 1 and needs a 2D"
            << " mesh, but the mesh has dimension " << dim_
            << exit(FatalError);
    }
    if (problem_ == problemType::example2 && dim_ != 3)
    {
        FatalErrorInFunction
            << "solution example2 is the 3D problem of Example 2 and needs a 3D"
            << " mesh, but the mesh has dimension " << dim_
            << exit(FatalError);
    }
}


// ************************************************************************* //
