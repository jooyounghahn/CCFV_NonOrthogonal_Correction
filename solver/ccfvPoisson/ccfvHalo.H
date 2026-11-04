/*---------------------------------------------------------------------------*\
    ccfvPoisson: cell-centered finite volume method with non-orthogonal
    correction for the Poisson problem on polytopal meshes

    Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
    SPDX-License-Identifier: GPL-3.0-or-later

    Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
-------------------------------------------------------------------------------
Class
    Foam::ccfvHalo

Description
    Sorts the patches of the mesh and exchanges cell values across the
    processor patches of a decomposed mesh (one layer of halo cells).

    Patch kinds:
      - processor patch: its faces are interior facets of the global mesh;
        the second cell of each facet lies on the neighbouring rank;
      - empty patch: front or back of a 2D mesh that is one cell thick;
        these faces are not facets of the 2D problem;
      - any other patch: boundary facets, which carry the Dirichlet data.
    Other constraint patches (cyclic, symmetry, wedge, ...) are rejected.

    exchange() returns, for every face of every processor patch, the value
    of the cell on the other side of the face.  Face i of a processor patch
    matches face i of the corresponding patch on the neighbouring rank, as
    written by decomposePar.  At most one processor patch per neighbouring
    rank is accepted, so that the messages have a unique order.  The values
    travel in binary form and arrive unchanged.

    In a serial run there are no processor patches and exchange() returns
    empty lists.

\*---------------------------------------------------------------------------*/

#ifndef ccfvHalo_H
#define ccfvHalo_H

#include "fvMesh.H"
#include "processorPolyPatch.H"
#include "emptyPolyPatch.H"
#include "PstreamBuffers.H"
#include "UIndirectList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                          Class ccfvHalo Declaration
\*---------------------------------------------------------------------------*/

class ccfvHalo
{
    // Private Data

        //- The mesh (on this rank)
        const fvMesh& mesh_;

        //- Patch indices of the processor patches
        labelList processorPatches_;

        //- Patch indices of the boundary (Dirichlet) patches
        labelList dirichletPatches_;

        //- Patch indices of the empty patches of a 2D mesh
        labelList emptyPatches_;

        //- Neighbouring rank of each processor patch
        labelList neighbourRanks_;


public:

    // Constructors

        //- Sort the patches of the mesh
        explicit ccfvHalo(const fvMesh& mesh)
        :
            mesh_(mesh)
        {
            const polyBoundaryMesh& patches = mesh.boundaryMesh();

            DynamicList<label> processor;
            DynamicList<label> dirichlet;
            DynamicList<label> empty;
            DynamicList<label> ranks;

            forAll(patches, patchi)
            {
                const polyPatch& pp = patches[patchi];

                if (const auto* procPatch = isA<processorPolyPatch>(pp))
                {
                    if (!UPstream::parRun())
                    {
                        FatalErrorInFunction
                            << "Processor patch " << pp.name()
                            << " found in a serial run" << exit(FatalError);
                    }

                    const label rank = procPatch->neighbProcNo();
                    if (ranks.contains(rank))
                    {
                        FatalErrorInFunction
                            << "Two processor patches to rank " << rank
                            << " (only one per neighbour is supported)"
                            << exit(FatalError);
                    }
                    processor.push_back(patchi);
                    ranks.push_back(rank);
                }
                else if (isA<emptyPolyPatch>(pp))
                {
                    empty.push_back(patchi);
                }
                else if (polyPatch::constraintType(pp.type()) || pp.coupled())
                {
                    FatalErrorInFunction
                        << "Patch " << pp.name() << " of type " << pp.type()
                        << " is not supported (allowed: processor, empty and"
                        << " plain boundary patches)" << exit(FatalError);
                }
                else
                {
                    dirichlet.push_back(patchi);
                }
            }

            processorPatches_.transfer(processor);
            dirichletPatches_.transfer(dirichlet);
            emptyPatches_.transfer(empty);
            neighbourRanks_.transfer(ranks);
        }


    // Member Functions

        //- The mesh
        const fvMesh& mesh() const noexcept
        {
            return mesh_;
        }

        //- Patch indices of the processor patches
        const labelList& processorPatches() const noexcept
        {
            return processorPatches_;
        }

        //- Patch indices of the boundary (Dirichlet) patches
        const labelList& dirichletPatches() const noexcept
        {
            return dirichletPatches_;
        }

        //- Patch indices of the empty patches
        const labelList& emptyPatches() const noexcept
        {
            return emptyPatches_;
        }

        //- Values of the cells on the other side of the processor faces,
        //- indexed [patchi][facei]; the lists of other patches are empty
        template<class Type>
        List<Field<Type>> exchange(const UList<Type>& cellValues) const
        {
            const polyBoundaryMesh& patches = mesh_.boundaryMesh();

            List<Field<Type>> haloValues(patches.size());

            if (processorPatches_.empty())
            {
                return haloValues;
            }

            PstreamBuffers buffers(UPstream::commsTypes::nonBlocking);

            // Send the values of the cells next to each processor patch
            for (const label patchi : processorPatches_)
            {
                const auto& procPatch =
                    refCast<const processorPolyPatch>(patches[patchi]);

                UOPstream toNeighbour(procPatch.neighbProcNo(), buffers);
                toNeighbour
                    << Field<Type>
                       (
                           UIndirectList<Type>
                           (
                               cellValues,
                               procPatch.faceCells()
                           )
                       );
            }

            buffers.finishedNeighbourSends(neighbourRanks_);

            // Receive the values of the cells on the other side
            for (const label patchi : processorPatches_)
            {
                const auto& procPatch =
                    refCast<const processorPolyPatch>(patches[patchi]);

                UIPstream fromNeighbour(procPatch.neighbProcNo(), buffers);
                Field<Type> received(fromNeighbour);

                if (received.size() != procPatch.size())
                {
                    FatalErrorInFunction
                        << "Patch " << procPatch.name() << ": received "
                        << received.size() << " values for "
                        << procPatch.size() << " faces" << exit(FatalError);
                }

                haloValues[patchi].transfer(received);
            }

            return haloValues;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
