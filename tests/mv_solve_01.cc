
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  // setup system
  const unsigned int dim           = 2;
  const unsigned int fe_degree     = 3;
  const unsigned int n_refinements = 3;

  MappingQ1<dim> mapping;
  FE_Q<dim>      fe(fe_degree);
  QGauss<dim>    quadrature(fe_degree + 1);

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  const unsigned int n_dofs = dof_handler.n_dofs();

  std::cout << n_dofs << std::endl;

  // create matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern(n_dofs, n_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraints);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix sparse_matrix(sparsity_pattern);

  MatrixCreator::create_laplace_matrix<dim, dim>(
    mapping, dof_handler, quadrature, sparse_matrix, nullptr, constraints);

  // create preconditioner
  TrilinosWrappers::PreconditionILU ilu;
  ilu.initialize(sparse_matrix);

  const unsigned int n_blocks        = n_dofs;
  const unsigned int n_blocks_stride = n_blocks;

  std::vector<Vector<double>> rhs(n_blocks);
  std::vector<Vector<double>> solution(n_blocks);

  for (unsigned int b = 0; b < n_blocks; ++b)
    {
      rhs[b].reinit(n_dofs);
      solution[b].reinit(n_dofs);
    }

  for (unsigned int b = 0; b < n_blocks; ++b)
    rhs[b][b] = 1.0;

  for (unsigned int b = 0; b < n_blocks; b += n_blocks_stride)
    {
      const unsigned int bend = std::min(n_blocks, b + n_blocks_stride);

      std::vector<double> rhs_temp(n_dofs * (bend - b));
      std::vector<double> solution_temp(n_dofs * (bend - b));

      for (unsigned int i = 0; i < (bend - b); ++i)
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            rhs_temp[i * n_dofs + j]      = rhs[i + b][j];
            solution_temp[i * n_dofs + j] = solution[i + b][j];
          }

      std::vector<double *> rhs_ptrs(bend - b);
      std::vector<double *> sultion_ptrs(bend - b);

      for (unsigned int i = 0; i < (bend - b); ++i)
        {
          rhs_ptrs[i]     = &rhs_temp[i * n_dofs];      //&rhs[i + b][0];
          sultion_ptrs[i] = &solution_temp[i * n_dofs]; //&solution[i + b][0];
        }

      const Epetra_CrsMatrix &mat  = sparse_matrix.trilinos_matrix();
      const Epetra_Operator  &prec = ilu.trilinos_operator();

      Epetra_MultiVector trilinos_dst(View,
                                      mat.OperatorRangeMap(),
                                      sultion_ptrs.data(),
                                      sultion_ptrs.size());
      Epetra_MultiVector trilinos_src(View,
                                      mat.OperatorDomainMap(),
                                      rhs_ptrs.data(),
                                      rhs_ptrs.size());

      ReductionControl solver_control;

      if (false)
        {
          TrilinosWrappers::SolverCG solver(solver_control);
          solver.solve(mat, trilinos_dst, trilinos_src, prec);
        }
      else
        {
          TrilinosWrappers::SolverDirect solver(solver_control);
          solver.initialize(sparse_matrix);
          solver.solve(mat, trilinos_dst, trilinos_src);
        }

      for (unsigned int i = 0; i < (bend - b); ++i)
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            solution[i + b][j] = solution_temp[i * n_dofs + j];
          }
    }

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;

  DataOut<dim> data_out;
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);
  for (unsigned int b = 0; b < n_blocks; ++b)
    data_out.add_data_vector(solution[b], "solution_" + std::to_string(b));
  data_out.build_patches();
  const std::string file_name = "solution.vtu";
  std::ofstream     file(file_name);
  data_out.write_vtu(file);
}
