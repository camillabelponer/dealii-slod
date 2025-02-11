// LOD as triple-matrix multiplication:
//   A^LOD = C^T A^FEM C
//
// implemented as
//
// | A^LOD  0 |
// |   0    0 |
//                =
// | 0  C^T | | 0    0   | | 0 0 |
// | 0   0  | | 0  A^FEM | | C 0 |
//
// Needed: C (computed column-wise, for AffineConstraints
// we need to access the full row)

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q_iso_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

using namespace dealii;

int
main()
{
  // 1) initialize system
  const unsigned int dim            = 1;
  const unsigned int n_subdivisions = 6;
  const unsigned int fe_degree      = 4;

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, n_subdivisions);

  DoFHandler<dim> dof_handler_fine(tria);
  dof_handler_fine.distribute_dofs(FE_Q_iso_Q1<dim>(fe_degree));

  DoFHandler<dim> dof_handler_coarse(tria);
  dof_handler_coarse.distribute_dofs(FE_DGQ<dim>(0));

  // 2) ininitialize sparsity pattern (TODO: multiple layers)
  DynamicSparsityPattern dsp_A_lod(dof_handler_coarse.n_dofs());
  DynamicSparsityPattern dsp_C(dof_handler_fine.n_dofs(),
                               dof_handler_coarse.n_dofs());

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned()) // parallel for-loop
      {
        std::vector<types::global_dof_index> local_dof_indices_coarse;

        if (!cell->at_boundary(0))
          local_dof_indices_coarse.emplace_back(cell->active_cell_index() - 1);
        local_dof_indices_coarse.emplace_back(cell->active_cell_index());
        if (!cell->at_boundary(1))
          local_dof_indices_coarse.emplace_back(cell->active_cell_index() + 1);

        const auto cell_fine = cell->as_dof_handler_iterator(dof_handler_fine);
        std::vector<types::global_dof_index> local_dof_indices_fine(
          cell_fine->get_fe().n_dofs_per_cell());
        cell_fine->get_dof_indices(local_dof_indices_fine);

        for (const auto &row_index : local_dof_indices_coarse)
          dsp_A_lod.add_row_entries(row_index, local_dof_indices_coarse);

        for (const auto &row_index : local_dof_indices_fine)
          dsp_C.add_row_entries(
            row_index,
            std::vector<types::global_dof_index>(1, cell->active_cell_index()));
      }

  // 3) initialize matrices
  SparsityPattern      sparsity_pattern_A_lod; // TODO: Trilinos
  SparseMatrix<double> A_lod;                  //
  SparsityPattern      sparsity_pattern_C;     //
  SparseMatrix<double> C;                      //

  sparsity_pattern_A_lod.copy_from(dsp_A_lod);
  A_lod.reinit(sparsity_pattern_A_lod);

  sparsity_pattern_C.copy_from(dsp_C);
  C.reinit(sparsity_pattern_C);

  // 4) set dummy constraints (TODO: adjust for LOD)
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned()) // parallel for-loop
      {
        const auto j = cell->active_cell_index();

        const auto cell_fine = cell->as_dof_handler_iterator(dof_handler_fine);
        std::vector<types::global_dof_index> local_dof_indices_fine(
          cell_fine->get_fe().n_dofs_per_cell());
        cell_fine->get_dof_indices(local_dof_indices_fine);

        for (const auto i : local_dof_indices_fine)
          C.set(i, j, 1.0 /*TODO*/);
      }

  // 5) convert sparse matrix C to shifted AffineConstraints
  // (TODO: only loop over local rows)
  AffineConstraints<double> constraints_lod_fem;
  for (unsigned int row = 0; row < C.m(); ++row) // parallel for-loop
    {
      std::vector<std::pair<types::global_dof_index, double>> dependencies;

      for (auto entry = C.begin(row); entry != C.end(row); ++entry)
        dependencies.emplace_back(entry->column(), entry->value());

      constraints_lod_fem.add_constraint(row + dof_handler_coarse.n_dofs(),
                                         dependencies);
    }
  // TODO: use AffineConstraints::make_consistent_in_parallel()
  constraints_lod_fem.close();

  // 6) assembly LOD matrix
  Vector<double> rhs_lod(dof_handler_coarse.n_dofs());
  for (const auto &cell : dof_handler_fine.active_cell_iterators())
    if (cell->is_locally_owned()) // parallel for-loop
      {
        const unsigned int n_dofs_per_cell = cell->get_fe().n_dofs_per_cell();

        // a) compute FEM element stiffness matrix
        FullMatrix<double> cell_matrix_fem(n_dofs_per_cell, n_dofs_per_cell);
        Vector<double>     cell_vector_fem(n_dofs_per_cell);

        for (unsigned int i = 0; i < cell_matrix_fem.m(); ++i)
          for (unsigned int j = 0; j < cell_matrix_fem.n(); ++j)
            cell_matrix_fem[i][j] = 1.0; // TODO: element stiffeness matrix

        for (unsigned int i = 0; i < cell_vector_fem.size(); ++i)
          cell_vector_fem[i] = 1.0;

        // b) assemble into LOD matrix by using constraints
        std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (auto &i : local_dof_indices)
          i += dof_handler_coarse.n_dofs(); // shifted view

        constraints_lod_fem.distribute_local_to_global(cell_matrix_fem,
                                                       local_dof_indices,
                                                       local_dof_indices,
                                                       A_lod);

        constraints_lod_fem.distribute_local_to_global(cell_vector_fem,
                                                       local_dof_indices,
                                                       rhs_lod);
      }

  A_lod.print(std::cout);

  // 7) solve
}