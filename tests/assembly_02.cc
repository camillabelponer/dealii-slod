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



void
my_Gauss_elimination(const FullMatrix<double> &            rhs,
                  const TrilinosWrappers::SparseMatrix &sparse_matrix,
                  FullMatrix<double> &                  solution,
                  double                                reduce    = 1.e-16,
                  double                                tolerance = 1.e-18,
                  double                                iter      = 100)
{
  // create preconditioner
  TrilinosWrappers::PreconditionILU ilu;
  ilu.initialize(sparse_matrix);

  Assert(sparse_matrix.m() == sparse_matrix.n(), ExcInternalError());
  Assert(rhs.m() == sparse_matrix.m(), ExcInternalError());
  Assert(rhs.m() == solution.m(), ExcInternalError());
  Assert(rhs.n() == solution.n(), ExcInternalError());

  solution = 0.0;

  const unsigned int n_dofs       = rhs.m();
  const unsigned int Ndofs_coarse = rhs.n();

  const unsigned int n_blocks        = Ndofs_coarse;
  const unsigned int n_blocks_stride = n_blocks;


  for (unsigned int b = 0; b < n_blocks; b += n_blocks_stride)
    {
      const unsigned int bend = std::min(n_blocks, b + n_blocks_stride);

      std::vector<double> rhs_temp(n_dofs * (bend - b));
      std::vector<double> solution_temp(n_dofs * (bend - b));

      for (unsigned int i = 0; i < (bend - b); ++i)
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            rhs_temp[i * n_dofs + j] = rhs(j, i + b); // rhs[i + b][j];
            solution_temp[i * n_dofs + j] =
              0.0; // solution(i+b,j); //solution[i + b][j];
          }

      std::vector<double *> rhs_ptrs(bend - b);
      std::vector<double *> sultion_ptrs(bend - b);

      for (unsigned int i = 0; i < (bend - b); ++i)
        {
          rhs_ptrs[i]     = &rhs_temp[i * n_dofs];      //&rhs[i + b][0];
          sultion_ptrs[i] = &solution_temp[i * n_dofs]; //&solution[i + b][0];
        }

      const Epetra_CrsMatrix &mat  = sparse_matrix.trilinos_matrix();
      const Epetra_Operator & prec = ilu.trilinos_operator();

      Epetra_MultiVector trilinos_dst(View,
                                      mat.OperatorRangeMap(),
                                      sultion_ptrs.data(),
                                      sultion_ptrs.size());
      Epetra_MultiVector trilinos_src(View,
                                      mat.OperatorDomainMap(),
                                      rhs_ptrs.data(),
                                      rhs_ptrs.size());


      if (false)
        {
          ReductionControl solver_control(
            iter, tolerance, reduce, false, false);
          TrilinosWrappers::SolverCG solver(solver_control);
          solver.solve(mat, trilinos_dst, trilinos_src, prec);
        }
      else
        {
          SolverControl solver_control(iter, tolerance, false, false);
          TrilinosWrappers::MySolverDirect solver(solver_control);
          solver.initialize(sparse_matrix);
          solver.solve(mat, trilinos_dst, trilinos_src);
        }

      for (unsigned int i = 0; i < (bend - b); ++i)
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            solution(j, i + b) = solution_temp[i * n_dofs + j];
          }
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim            = 2;
  const unsigned int fe_degree      = 2;
  const unsigned int n_overlap      = 1; // numbers::invalid_unsigned_int
  const unsigned int n_subdivisions = 2;
  const MPI_Comm     comm           = MPI_COMM_WORLD;

  AssertDimension(Utilities::MPI::n_mpi_processes(comm), 1);

  std::vector<unsigned int> repetitions(dim, n_subdivisions);
  Point<dim>                p1;
  Point<dim>                p2;

  for (unsigned int d = 0; d < dim; ++d)
    p2[d] = 1.0;

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
        patch.reinit(cell, n_overlap * 2);
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

  // 4) set dummy constraints
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned()) // parallel for-loop
      {
        patch.reinit(cell, n_overlap);

        double H = 1.0 / n_subdivisions;
        double h = H / fe_degree;

        const auto                           n_dofs_patch  = patch.n_dofs();
        const unsigned int                   N_dofs_coarse = patch.n_cells();
        const unsigned int                   N_dofs_fine   = n_dofs_patch;
        std::vector<types::global_dof_index> local_dof_indices_fine(
          n_dofs_patch);
        patch.get_dof_indices(local_dof_indices_fine);

        AffineConstraints<double> patch_constraints;
        for (unsigned int d = 0; d < 2 * dim; ++d)
          patch.make_zero_boundary_constraints<double>(d, patch_constraints);
        patch_constraints.close();

        Vector<double> selected_basis_function(n_dofs_patch);

        TrilinosWrappers::SparsityPattern sparsity_pattern(n_dofs_patch,
                                                           n_dofs_patch);
        patch.create_sparsity_pattern(patch_constraints, sparsity_pattern);
        sparsity_pattern.compress();

        TrilinosWrappers::SparseMatrix patch_stiffness_matrix(sparsity_pattern);
        FullMatrix<double>             PT(N_dofs_fine, N_dofs_coarse);
        FullMatrix<double>             P_Ainv_PT(N_dofs_coarse);
        FullMatrix<double>             Ainv_PT(N_dofs_fine, N_dofs_coarse);

        Vector<double> PT_counter(N_dofs_fine);

        FE_Q_iso_Q1<dim>     fe(fe_degree);
        const QIterated<dim> quadrature(QGauss<1>(2), fe_degree);
        FEValues<dim>        fe_values(
          fe, quadrature, update_values | update_gradients | update_JxW_values);

        // ... by looping over cells in patch
        for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
          {
            fe_values.reinit(patch.create_cell_iterator(tria, cell));

            const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

            FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

            for (const unsigned int q_index :
                 fe_values.quadrature_point_indices())
              {
                for (const unsigned int i : fe_values.dof_indices())
                  for (const unsigned int j : fe_values.dof_indices())
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                                          fe_values.shape_grad(j, q_index) *
                                          fe_values.JxW(q_index));
              }

            std::vector<types::global_dof_index> indices(dofs_per_cell);
            patch.get_dof_indices_of_cell(cell, indices);

            AffineConstraints<double>().distribute_local_to_global(
              cell_matrix, indices, patch_stiffness_matrix);

            for (const auto i : indices)
              {
                PT[i][cell] = h * h;
                PT_counter[i] += 1;
              }
          }

        patch_stiffness_matrix.compress(VectorOperation::values::add);

        for (auto &i : PT_counter)
          i = 1.0 / i;

        for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
          for (unsigned int i = 0; i < n_dofs_patch; ++i)
            PT[i][cell] *= PT_counter[i];

        IndexSet patch_constraints_is(n_dofs_patch);
        for(const auto & l : patch_constraints.get_lines())
          patch_constraints_is.add_index(l.index);

        for (unsigned int i = 0; i < patch.n_cells(); ++i)
          for (const auto j : patch_constraints_is)
            PT(j, i) = 0.0;

        for (const auto j : patch_constraints_is)
          patch_stiffness_matrix.clear_row(j, 1);

        my_Gauss_elimination(PT, patch_stiffness_matrix, Ainv_PT);

        PT.Tmmult(P_Ainv_PT, Ainv_PT);
        P_Ainv_PT /= pow(H, dim);
        P_Ainv_PT.gauss_jordan();

        Vector<double> e_i(N_dofs_coarse);
        Vector<double> triple_product_inv_e_i(N_dofs_coarse);

        e_i[patch.cell_index(cell)] = 1.0;
        P_Ainv_PT.vmult(triple_product_inv_e_i, e_i);

        Ainv_PT.vmult(selected_basis_function, triple_product_inv_e_i);

        selected_basis_function /= selected_basis_function.l2_norm();

        patch_constraints.set_zero(selected_basis_function);

        for (unsigned int i = 0; i < n_dofs_patch; ++i)
          if (selected_basis_function[i] != 0.0)
            C.set(local_dof_indices_fine[i],
                  cell->active_cell_index(),
                  selected_basis_function[i]);
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

  LinearAlgebra::distributed::Vector<double> rhs_lod(
    n_dofs_coarse); // TODO: parallel

  // 6) assembly LOD matrix
  FE_Q_iso_Q1<dim>     fe(fe_degree);
  const QIterated<dim> quadrature(QGauss<1>(2), fe_degree);
  FEValues<dim>        fe_values(fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values);

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned()) // parallel for-loop
      {
        Patch<dim> patch(fe_degree, repetitions);
        patch.reinit(cell, 0);

        fe_values.reinit(cell);

        const unsigned int n_dofs_per_cell = patch.n_dofs();

        // a) compute FEM element stiffness matrix
        FullMatrix<double> cell_matrix_fem(n_dofs_per_cell, n_dofs_per_cell);
        Vector<double>     cell_rhs_fem(n_dofs_per_cell);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            for (const unsigned int i : fe_values.dof_indices())
              for (const unsigned int j : fe_values.dof_indices())
                cell_matrix_fem(i, j) +=
                  (fe_values.shape_grad(i, q_index) *
                   fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));

            for (const unsigned int i : fe_values.dof_indices())
              cell_rhs_fem(i) += (fe_values.shape_value(i, q_index) * 1. *
                                  fe_values.JxW(q_index));
          }

        // b) assemble into LOD matrix by using constraints
        std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
        patch.get_dof_indices(local_dof_indices, true /*hiearchical*/);

        for (auto &i : local_dof_indices)
          i += n_dofs_coarse; // shifted view

        constraints_lod_fem.distribute_local_to_global(cell_matrix_fem,
                                                       local_dof_indices,
                                                       local_dof_indices,
                                                       A_lod);

        constraints_lod_fem.distribute_local_to_global(cell_rhs_fem,
                                                       local_dof_indices,
                                                       rhs_lod);
      }

  A_lod.compress(VectorOperation::values::add);
  rhs_lod.compress(VectorOperation::values::add);

  std::cout << A_lod.frobenius_norm() << std::endl; // TODO

  // 7) solve LOD system
  LinearAlgebra::distributed::Vector<double> solution_lod;
  solution_lod.reinit(rhs_lod);

  TrilinosWrappers::SolverDirect solver;
  solver.solve(A_lod, solution_lod, rhs_lod);

  rhs_lod.print(std::cout);      // TODO
  solution_lod.print(std::cout); // TODO

  // 8) convert to FEM solution
  LinearAlgebra::distributed::Vector<double> solution_fem(
    n_dofs_fine); // TODO: parallel

  for (const auto i : locally_owned_fine_dofs)
    if (const auto constraint_entries =
          constraints_lod_fem.get_constraint_entries(i + n_dofs_coarse))
      for (const auto &[j, weight] : *constraint_entries)
        solution_fem[i] += weight * solution_lod[j];

  // 8) output LOD and FEM results

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  compute_renumbering_lex(dof_handler);

  MappingQ<dim> mapping(1);

  DataOutBase::VtkFlags flags;

  if (dim > 1)
    flags.write_higher_order_cells = true;

  DataOut<dim> data_out;
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(solution_lod, "solution_lod");
  data_out.add_data_vector(solution_fem, "solution_fem");

  data_out.build_patches(mapping, fe_degree);

  const std::string file_name = "solution.vtu";

  std::ofstream file(file_name);
  data_out.write_vtu(file);
}