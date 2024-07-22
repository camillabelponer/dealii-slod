#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>

#include "util.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim       = 2;
  const unsigned int fe_degree = 1;
  const unsigned int n_overlap = 1; // numbers::invalid_unsigned_int
  const MPI_Comm     comm      = MPI_COMM_WORLD;

  std::vector<unsigned int> repetitions = {{5, 5}};
  Point<dim>                p1(0, 0);
  Point<dim>                p2(1, 1);

  parallel::shared::Triangulation<dim> tria(comm);
  GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);

  types::global_dof_index n_dofs_coarse = 1;
  types::global_dof_index n_dofs_fine   = 1;
  for (unsigned int d = 0; d < dim; ++d)
    {
      n_dofs_coarse *= repetitions[d];
      n_dofs_fine *= repetitions[d] * fe_degree + 1;
    }

  AssertDimension(n_dofs_coarse, tria.n_active_cells());

  const auto locally_owned_fine_dofs =
    Utilities::create_evenly_distributed_partitioning(
      Utilities::MPI::this_mpi_process(comm),
      Utilities::MPI::n_mpi_processes(comm),
      n_dofs_fine);

  Patch<dim> patch(fe_degree, repetitions);

  IndexSet locally_owned_cells(n_dofs_coarse);

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      locally_owned_cells.add_index(cell->active_cell_index());

  // 2) ininitialize sparsity pattern
  TrilinosWrappers::SparsityPattern sparsity_pattern_A_lod(locally_owned_cells,
                                                           comm);

  TrilinosWrappers::SparsityPattern sparsity_pattern_C(locally_owned_fine_dofs,
                                                       comm);

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned()) // parallel for-loop
      {
        // A_lod sparsity pattern
        patch.reinit(cell, n_overlap + 1 /*TODO: ?*/);
        std::vector<types::global_dof_index> local_dof_indices_coarse;
        for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
          local_dof_indices_coarse.emplace_back(
            patch.create_cell_iterator(tria, cell)->active_cell_index());

        for (const auto &row_index : local_dof_indices_coarse)
          sparsity_pattern_A_lod.add_row_entries(row_index,
                                                 local_dof_indices_coarse);

        // C sparsity pattern
        patch.reinit(cell, n_overlap);
        const auto                           n_dofs_patch = patch.n_dofs();
        std::vector<types::global_dof_index> local_dof_indices_fine(
          n_dofs_patch);
        patch.get_dof_indices(local_dof_indices_fine);

        AffineConstraints<double> patch_constraints;
        for (unsigned int d = 0; d < 2 * dim; ++d)
          patch.make_zero_boundary_constraints<double>(d, patch_constraints);
        patch_constraints.close();

        for (unsigned int i = 0; i < n_dofs_patch; ++i)
          if (!patch_constraints.is_constrained(i))
            sparsity_pattern_C.add_row_entries(
              local_dof_indices_fine[i],
              std::vector<types::global_dof_index>(1,
                                                   cell->active_cell_index()));
      }

  sparsity_pattern_A_lod.compress();
  sparsity_pattern_C.compress();

  // 3) initialize matrices
  TrilinosWrappers::SparseMatrix A_lod(sparsity_pattern_A_lod);
  TrilinosWrappers::SparseMatrix C(sparsity_pattern_C);

  // 4) set dummy constraints (TODO: adjust for LOD)
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned()) // parallel for-loop
      {
        patch.reinit(cell, n_overlap);

        const auto                           n_dofs_patch = patch.n_dofs();
        std::vector<types::global_dof_index> local_dof_indices_fine(
          n_dofs_patch);
        patch.get_dof_indices(local_dof_indices_fine);

        AffineConstraints<double> patch_constraints;
        for (unsigned int d = 0; d < 2 * dim; ++d)
          patch.make_zero_boundary_constraints<double>(d, patch_constraints);
        patch_constraints.close();

        for (unsigned int i = 0; i < n_dofs_patch; ++i)
          if (!patch_constraints.is_constrained(i))
            C.set(local_dof_indices_fine[i],
                  cell->active_cell_index(),
                  1.0 /*TODO*/);
      }

  C.compress(VectorOperation::values::insert);

  // 5) convert sparse matrix C to shifted AffineConstraints
  IndexSet constraints_lod_fem_locally_owned_dofs(n_dofs_fine + n_dofs_coarse);
  constraints_lod_fem_locally_owned_dofs.add_indices(locally_owned_cells);
  constraints_lod_fem_locally_owned_dofs.add_indices(locally_owned_fine_dofs,
                                                     n_dofs_coarse);

  IndexSet constraints_lod_fem_locally_stored_constraints =
    constraints_lod_fem_locally_owned_dofs;

  for (const auto row : locally_owned_fine_dofs) // parallel for-loop
    {
      for (auto entry = C.begin(row); entry != C.end(row); ++entry)
        constraints_lod_fem_locally_stored_constraints.add_index(
          entry->column()); // coarse
    }

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned()) // parallel for-loop
      {
        patch.reinit(cell, n_overlap);

        const auto                           n_dofs_patch = patch.n_dofs();
        std::vector<types::global_dof_index> local_dof_indices_fine(
          n_dofs_patch);
        patch.get_dof_indices(local_dof_indices_fine);

        for (unsigned int i = 0; i < n_dofs_patch; ++i)
          constraints_lod_fem_locally_stored_constraints.add_index(
            local_dof_indices_fine[i] + n_dofs_coarse); // fine
      }

  AffineConstraints<double> constraints_lod_fem(
    constraints_lod_fem_locally_owned_dofs,
    constraints_lod_fem_locally_stored_constraints);
  for (const auto row : locally_owned_fine_dofs) // parallel for-loop
    {
      std::vector<std::pair<types::global_dof_index, double>> dependencies;

      for (auto entry = C.begin(row); entry != C.end(row); ++entry)
        dependencies.emplace_back(entry->column(), entry->value());

      if (true || !dependencies.empty())
        constraints_lod_fem.add_constraint(row + n_dofs_coarse, dependencies);
    }

  constraints_lod_fem.make_consistent_in_parallel(
    constraints_lod_fem_locally_owned_dofs,
    constraints_lod_fem_locally_stored_constraints,
    comm);
  constraints_lod_fem.close();

  // 6) assembly LOD matrix
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned()) // parallel for-loop
      {
        Patch<dim> patch(fe_degree, repetitions);
        patch.reinit(cell, 1);

        const unsigned int n_dofs_per_cell = patch.n_dofs();

        // a) compute FEM element stiffness matrix
        FullMatrix<double> cell_matrix_fem(n_dofs_per_cell, n_dofs_per_cell);

        for (unsigned int i = 0; i < cell_matrix_fem.m(); ++i)
          for (unsigned int j = 0; j < cell_matrix_fem.n(); ++j)
            cell_matrix_fem[i][j] = 1.0; // TODO: element stiffeness matrix

        // b) assemble into LOD matrix by using constraints
        std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
        patch.get_dof_indices(local_dof_indices);

        for (auto &i : local_dof_indices)
          i += n_dofs_coarse; // shifted view

        constraints_lod_fem.distribute_local_to_global(cell_matrix_fem,
                                                       local_dof_indices,
                                                       local_dof_indices,
                                                       A_lod);
      }
}
