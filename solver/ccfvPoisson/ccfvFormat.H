/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
-------------------------------------------------------------------------------
Description
    Number format of all floating-point values in the solver output:
    scientific notation with ten digits after the decimal point, the same
    as the C format "%.10e".

\*---------------------------------------------------------------------------*/

#ifndef ccfvFormat_H
#define ccfvFormat_H

#include "word.H"
#include "scalar.H"

#include <iomanip>
#include <sstream>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

//- The value x in the format %.10e.  The text is returned as a word, which
//- an Ostream writes without quotes.
inline word ccfvSci(const scalar x)
{
    std::ostringstream text;
    text << std::scientific << std::setprecision(10) << x;
    return word(text.str(), false);
}

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
