/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
\*---------------------------------------------------------------------------*/

#include "ccfvScheme.H"

// * * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * //

const Foam::Enum<Foam::ccfvScheme::correctionType>
Foam::ccfvScheme::correctionNames
({
    { correctionType::minimum, "minimum" },
    { correctionType::fullArea, "fullArea" },
    { correctionType::overRelaxed, "overRelaxed" },
});


const Foam::Enum<Foam::ccfvScheme::weightChoice>
Foam::ccfvScheme::weightNames
({
    { weightChoice::i, "i" },
    { weightChoice::ii, "ii" },
    { weightChoice::iii, "iii" },
});


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::ccfvScheme::ccfvScheme(const dictionary& dict)
:
    correction_(correctionNames.get("correction", dict)),
    lsWeights_(weightNames.get("lsWeights", dict)),
    averageWeights_(weightNames.get("averageWeights", dict))
{}


// ************************************************************************* //
