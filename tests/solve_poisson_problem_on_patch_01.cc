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

  // 0) parameters
  const unsigned int dim       = 2;
  const unsigned int fe_degree = 7;
  const unsigned int n_overlap = 3; // numbers::invalid_unsigned_int

  std::vector<unsigned int> repetitions = {{10, 10}};
  Point<dim>                p1(0, 0);
  Point<dim>                p2(1, 1);

  // 1) define patch (parameter)
  std::array<unsigned int, dim> patch_start = {{1, 4}};
  std::array<unsigned int, dim> patch_size  = {{4, 3}};

  // 2) create mesh
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);

  // 3) create patch
  Patch<dim> patch(fe_degree, repetitions);

  if (n_overlap == numbers::invalid_unsigned_int)
    patch.reinit(patch_start, patch_size);
  else
    patch.reinit(tria.create_cell_iterator(CellId(
                   patch_start[0] + patch_start[1] * repetitions[0], {})),
                 n_overlap);


  // 4) make constraints on patch
  AffineConstraints<double> patch_constraints;
  for (unsigned int d = 0; d < 2 * dim; ++d)
    patch.make_zero_boundary_constraints<double>(d, patch_constraints);
  patch_constraints.close();

  // 5) assemble system on patch
  const auto n_dofs_patch = patch.n_dofs();

  TrilinosWrappers::SparsityPattern sparsity_pattern(n_dofs_patch,
                                                     n_dofs_patch);
  patch.create_sparsity_pattern(patch_constraints, sparsity_pattern);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix A(sparsity_pattern);
  Vector<double>                 rhs(n_dofs_patch);
  Vector<double>                 solution(n_dofs_patch);

  FE_Q_iso_Q1<dim>   fe(fe_degree);
  const QIterated<2> quadrature(QGauss<1>(2), fe_degree);
  FEValues<2>        fe_values(fe,
                        quadrature,
                        update_values | update_gradients | update_JxW_values);

  // ... by looping over cells in patch
  for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
    {
      fe_values.reinit(patch.create_cell_iterator(tria, cell));

      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) *
                 fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));

          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) +=
              (fe_values.shape_value(i, q_index) * 1. * fe_values.JxW(q_index));
        }

      std::vector<types::global_dof_index> indices(dofs_per_cell);
      patch.get_dof_indices_of_cell(cell, indices);

      patch_constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, indices, A, rhs);
    }

  A.compress(VectorOperation::values::add);

  // 6) solve patch system
  SolverControl                  sc;
  TrilinosWrappers::SolverDirect solver(sc);
  solver.solve(A, solution, rhs);

  // 7) visualization on fine mesh
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  compute_renumbering_lex(dof_handler);

  Vector<double> solution_fine(dof_handler.n_dofs());

  std::vector<types::global_dof_index> patch_indices(n_dofs_patch);
  patch.get_dof_indices(patch_indices);
  AffineConstraints<double>().distribute_local_to_global(solution,
                                                         patch_indices,
                                                         solution_fine);

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;

  MappingQ<dim> mapping(1);

  DataOut<dim> data_out;
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(solution_fine, "solution");

  data_out.build_patches(mapping, fe_degree);

  const std::string file_name = "solution.vtu";

  std::ofstream file(file_name);
  data_out.write_vtu(file);

  solution_fine.print(std::cout);
}
