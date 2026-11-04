/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
-------------------------------------------------------------------------------
Class
    Foam::ccfvScheme

Description
    The three choices that define the method of the paper, and the formulas
    that depend on them.  Read from the dictionary entries

        correction      minimum | fullArea | overRelaxed;   // c_F of the flux
        lsWeights       i | ii | iii;       // omega_{K,F}
        averageWeights  i | ii | iii;       // alpha_{K,F}

    Non-orthogonal correction of the flux:
        c_F = cos theta_F (minimum), 1 (full-area), sec theta_F (over-relaxed),
        e_{K,F} = n_{K,F} - c_F r_{K,F}.

    Weights of the least-squares gradient:
        (i)   omega_{K,F} = 1
        (ii)  omega_{K,F} = |F|
        (iii) omega_{K,F} = alpha_{K,F} d_F |F| / (|K| cos^2 theta_F)

    Weights of the facet average {{v}}_F = alpha_{K,F} v_K + alpha_{L,F} v_L
    on an interior facet F with the cells K and L:
        (i)   alpha_{K,F} = alpha_{L,F} = 1/2
        (ii)  alpha_{K,F} = d_{L,F}/d_F,  alpha_{L,F} = d_{K,F}/d_F
        (iii) alpha_{K,F} = d_{K,F}/d_F,  alpha_{L,F} = d_{L,F}/d_F
    and alpha_{K,F} = 1 on a boundary facet, where {{v}}_F = v_K.

    The least-squares weight (iii) contains alpha_{K,F}, so it follows the
    chosen average weights: alpha_{K,F} d_F = d_F/2 for (i), d_{L,F} for
    (ii), d_{K,F} for (iii), and d_{K,F} on a boundary facet.

    The convergence tables of the paper use overRelaxed with the weights
    (iii) and (iii).

SourceFiles
    ccfvScheme.C

\*---------------------------------------------------------------------------*/

#ifndef ccfvScheme_H
#define ccfvScheme_H

#include "dictionary.H"
#include "vector.H"
#include "Enum.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class ccfvScheme Declaration
\*---------------------------------------------------------------------------*/

class ccfvScheme
{
public:

    // Public Data Types

        //- The choice of c_F in the flux
        enum class correctionType { minimum, fullArea, overRelaxed };

        //- Choice (i), (ii) or (iii) of the least-squares or average weights
        enum class weightChoice { i, ii, iii };

        //- Dictionary names of the corrections
        static const Enum<correctionType> correctionNames;

        //- Dictionary names of the weight choices
        static const Enum<weightChoice> weightNames;


private:

    // Private Data

        correctionType correction_;

        weightChoice lsWeights_;

        weightChoice averageWeights_;


public:

    // Constructors

        //- Read the entries correction, lsWeights and averageWeights
        explicit ccfvScheme(const dictionary& dict);


    // Member Functions

        correctionType correction() const noexcept { return correction_; }

        weightChoice lsWeights() const noexcept { return lsWeights_; }

        weightChoice averageWeights() const noexcept
        {
            return averageWeights_;
        }

        const word& correctionName() const
        {
            return correctionNames.get(correction_);
        }

        const word& lsWeightsName() const
        {
            return weightNames.get(lsWeights_);
        }

        const word& averageWeightsName() const
        {
            return weightNames.get(averageWeights_);
        }


        //- The coefficient a_F = |F| c_F/delta_F of the two-point part of
        //- the flux, from S = |F| n_{K,F} and d = delta_F r_{K,F}
        //- (d = x_L - x_K, or x_F - x_K on a boundary facet), using
        //- cos theta_F = S.d/(|S| |d|):
        //-     minimum      a_F = S.d/(d.d)
        //-     fullArea     a_F = |S|/|d|
        //-     overRelaxed  a_F = S.S/(S.d)
        //- The corresponding |F| e_{K,F} is S - a_F d.
        scalar twoPointCoeff(const vector& S, const vector& d) const
        {
            switch (correction_)
            {
                case correctionType::minimum:     return (S & d)/(d & d);
                case correctionType::fullArea:    return mag(S)/mag(d);
                case correctionType::overRelaxed: return (S & S)/(S & d);
            }
            return 0;
        }

        //- alpha_{K,F} d_F of the average weights for the cell K of an interior
        //- facet, with dK = d_{K,F} and dL = d_{L,F} (d_F = dK + dL):
        //-     (i) d_F/2,   (ii) d_{L,F},   (iii) d_{K,F}.
        //- For the cell L call it with the two distances swapped.
        scalar alphaDF(const scalar dK, const scalar dL) const
        {
            switch (averageWeights_)
            {
                case weightChoice::i:   return 0.5*(dK + dL);
                case weightChoice::ii:  return dL;
                case weightChoice::iii: return dK;
            }
            return 0;
        }

        //- The facet average {{v}}_F = alpha_{K,F} v_K + alpha_{L,F} v_L of
        //- Average weights on an interior facet, dK = d_{K,F}, dL = d_{L,F}:
        //-     (i)   (v_K + v_L)/2
        //-     (ii)  (d_{L,F} v_K + d_{K,F} v_L)/d_F
        //-     (iii) (d_{K,F} v_K + d_{L,F} v_L)/d_F
        //- The value is the same when K and L exchange their roles.
        vector average
        (
            const vector& vK,
            const vector& vL,
            const scalar dK,
            const scalar dL
        ) const
        {
            switch (averageWeights_)
            {
                case weightChoice::i:   return 0.5*(vK + vL);
                case weightChoice::ii:  return (dL*vK + dK*vL)/(dK + dL);
                case weightChoice::iii: return (dK*vK + dL*vL)/(dK + dL);
            }
            return vector(Zero);
        }

        //- The least-squares weight omega_{K,F}:
        //-     (i) 1,   (ii) |F|,   (iii) alpha_{K,F} d_F |F|/(|K| cos^2 theta_F)
        //- with cos theta_F = d_F/delta_F.
        //- Arguments: alphaDF = alpha_{K,F} d_F (from alphaDF(), or d_{K,F}
        //- on a boundary facet), dF = d_F, deltaF = delta_F, magF = |F| and
        //- measK = |K|.
        scalar omega
        (
            const scalar alphaDF,
            const scalar dF,
            const scalar deltaF,
            const scalar magF,
            const scalar measK
        ) const
        {
            switch (lsWeights_)
            {
                case weightChoice::i:  return 1;
                case weightChoice::ii: return magF;
                case weightChoice::iii:
                {
                    const scalar cosTheta = dF/deltaF;
                    return alphaDF*magF/(measK*cosTheta*cosTheta);
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
