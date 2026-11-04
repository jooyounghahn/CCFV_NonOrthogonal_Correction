/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
-------------------------------------------------------------------------------
Class
    Foam::ccfvExactSolution

Description
    Exact solution phi and right-hand side f = -Laplace(phi) of the problem
    selected by the dictionary entry 'solution':

        example1 Example 1, d = 2:
                 phi = sin^2 x + cos^2 y + ((x + y)/sqrt 2)^3
                 f   = -2 cos 2x + 2 cos 2y - 6 (x + y)/sqrt 2
        example2 Example 2, d = 3:
                 phi = sin^2 x + cos^2 y + z^3
                 f   = -2 cos 2x + 2 cos 2y - 6 z
        linear   check of the implementation (not in the paper):
                 phi = 1 + 2x - 3y (d = 2),  1 + 2x - 3y + 4z (d = 3),  f = 0

    For an affine phi the least-squares gradient is exact, every facet
    average of it is exact, and the flux equals the exact flux for
    every c_F.  The discrete solution is then phi(x_K) for every
    choice of the correction and the weights, up to round-off and the
    stopping tolerance of the iteration.

SourceFiles
    ccfvExactSolution.C

\*---------------------------------------------------------------------------*/

#ifndef ccfvExactSolution_H
#define ccfvExactSolution_H

#include "vector.H"
#include "Enum.H"

#include <cmath>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class ccfvExactSolution Declaration
\*---------------------------------------------------------------------------*/

class ccfvExactSolution
{
public:

    // Public Data Types

        //- The test problems
        enum class problemType { example1, example2, linear };

        //- Dictionary names of the test problems
        static const Enum<problemType> problemNames;


private:

    // Private Data

        problemType problem_;

        //- Dimension of the problem (2 or 3)
        label dim_;


public:

    // Constructors

        //- Construct for a problem and the dimension of the mesh; stops
        //- with an error for example1 on a 3D mesh or example2 on a 2D mesh
        ccfvExactSolution(const problemType problem, const label dim);


    // Member Functions

        problemType problem() const noexcept { return problem_; }

        //- Dictionary name of the problem
        const word& name() const { return problemNames.get(problem_); }

        //- phi(x)
        scalar phi(const vector& x) const
        {
            switch (problem_)
            {
                case problemType::example1:
                {
                    const scalar s = std::sin(x.x());
                    const scalar c = std::cos(x.y());
                    const scalar w = (x.x() + x.y())/std::sqrt(2.0);
                    return s*s + c*c + w*w*w;
                }
                case problemType::example2:
                {
                    const scalar s = std::sin(x.x());
                    const scalar c = std::cos(x.y());
                    return s*s + c*c + x.z()*x.z()*x.z();
                }
                case problemType::linear:
                {
                    if (dim_ == 2)
                    {
                        return 1.0 + 2.0*x.x() - 3.0*x.y();
                    }
                    return 1.0 + 2.0*x.x() - 3.0*x.y() + 4.0*x.z();
                }
            }
            return 0;
        }

        //- f(x) = -Laplace(phi)(x)
        scalar f(const vector& x) const
        {
            switch (problem_)
            {
                case problemType::example1:
                {
                    const scalar w = (x.x() + x.y())/std::sqrt(2.0);
                    return
                       -(
                            2.0*std::cos(2.0*x.x()) - 2.0*std::cos(2.0*x.y())
                          + 6.0*w
                        );
                }
                case problemType::example2:
                {
                    return
                       -(
                            2.0*std::cos(2.0*x.x()) - 2.0*std::cos(2.0*x.y())
                          + 6.0*x.z()
                        );
                }
                case problemType::linear:
                {
                    return 0;
                }
            }
            return 0;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
