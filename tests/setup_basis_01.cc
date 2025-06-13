// Compute and plot a basis.

#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include "util.h"

using namespace dealii;

template <unsigned int dim>
void
test()
{
  const unsigned int n_oversampling      = 2;
  const unsigned int n_subdivisions_fine = 4;

  MappingQ1<dim>   mapping;
  FE_Q_iso_Q1<dim> fe(n_subdivisions_fine);
  QIterated<dim>   quadrature(QGauss<1>(2), n_subdivisions_fine);

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, 1 + 2 * n_oversampling);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  compute_renumbering_lex(dof_handler);

  AffineConstraints<double> constraints;
  constraints.close();

  Patch<dim> patch(n_subdivisions_fine,
                   std::vector<unsigned int>(dim, 3 + 2 * n_oversampling),
                   1);

  std::array<unsigned int, dim> patch_start;
  patch_start.fill(1);
  std::array<unsigned int, dim> patch_size;
  patch_size.fill(1 + 2 * n_oversampling);
  patch.reinit(patch_start, patch_size);

  TrilinosWrappers::SparsityPattern sparsity_pattern(dof_handler.n_dofs(),
                                                     dof_handler.n_dofs());
  patch.create_sparsity_pattern(constraints, sparsity_pattern);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix patch_stiffness_matrix(sparsity_pattern);


  FEValues<dim> fe_values(mapping,
                          fe,
                          quadrature,
                          update_gradients | update_JxW_values);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

      for (const auto q : fe_values.quadrature_point_indices())
        for (const auto i : fe_values.dof_indices())
          for (const auto j : fe_values.dof_indices())
            cell_matrix(i, j) +=
              (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) *
               fe_values.JxW(q));

      std::vector<types::global_dof_index> indices(dofs_per_cell);
      cell->get_dof_indices(indices);

      constraints.distribute_local_to_global(cell_matrix,
                                             indices,
                                             patch_stiffness_matrix);
    }
  patch_stiffness_matrix.compress(VectorOperation::values::add);

  LODPatchProblem<dim> lod_patch_problem(1, true, fe);

  unsigned int central_cell_id = 0;

  if (dim == 1)
    central_cell_id = n_oversampling;
  else if (dim == 2)
    central_cell_id =
      (1 + 2 * n_oversampling) * n_oversampling + n_oversampling;


  const auto selected_basis_function =
    lod_patch_problem.setup_basis(patch,
                                  central_cell_id,
                                  patch_stiffness_matrix);

  selected_basis_function[0].print(std::cout);

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;

  DataOut<dim> data_out;
  data_out.set_flags(flags);

  data_out.add_data_vector(dof_handler, selected_basis_function[0], "basis");

  data_out.build_patches(mapping, n_subdivisions_fine);

  data_out.write_vtu_in_parallel("selected_basis_" + std::to_string(dim) +
                                   ".vtu",
                                 tria.get_communicator());
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<1>();
  test<2>();
}