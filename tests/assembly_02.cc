// LOD as triple-matrix multiplication:
//   A^LOD = C^T A^FEM C
//
// implemented as block system
//
// | A^LOD  0 |
// |   0    0 |
//                =
// | 0  C^T | | 0    0   | | 0 0 |
// | 0   0  | | 0  A^FEM | | C 0 |
//
// via cell loop
//
// A^LOD = ∑ C_i^T A_i^FEM C_i
//
// Needed: C (computed column-wise, for AffineConstraints
// we need to access the full row)

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>

#include "util.h"

using namespace dealii;



namespace Step96
{

  void
  my_Gauss_elimination(const FullMatrix<double>             &rhs,
                       const TrilinosWrappers::SparseMatrix &sparse_matrix,
                       FullMatrix<double>                   &solution)
  {
    Assert(sparse_matrix.m() == sparse_matrix.n(), ExcInternalError());
    Assert(rhs.m() == sparse_matrix.m(), ExcInternalError());
    Assert(rhs.m() == solution.m(), ExcInternalError());
    Assert(rhs.n() == solution.n(), ExcInternalError());

    solution = 0.0;

    const unsigned int m = rhs.m();
    const unsigned int n = rhs.n();

    FullMatrix<double> rhs_t(n, m);
    FullMatrix<double> solution_t(n, m);

    rhs_t.copy_transposed(rhs);
    solution_t.copy_transposed(solution);

    std::vector<double *> rhs_ptrs(n);
    std::vector<double *> sultion_ptrs(n);

    for (unsigned int i = 0; i < n; ++i)
      {
        rhs_ptrs[i]     = &rhs_t[i][0];
        sultion_ptrs[i] = &solution_t[i][0];
      }

    const Epetra_CrsMatrix &mat = sparse_matrix.trilinos_matrix();

    Epetra_MultiVector trilinos_dst(View,
                                    mat.OperatorRangeMap(),
                                    sultion_ptrs.data(),
                                    sultion_ptrs.size());
    Epetra_MultiVector trilinos_src(View,
                                    mat.OperatorDomainMap(),
                                    rhs_ptrs.data(),
                                    rhs_ptrs.size());

    SolverControl                    solver_control(100, 1.e-18, false, false);
    TrilinosWrappers::MySolverDirect solver(solver_control);
    solver.initialize(sparse_matrix);
    solver.solve(mat, trilinos_dst, trilinos_src);

    solution.copy_transposed(solution_t);
  }

  struct Parameters
  {
    unsigned int n_subdivisions_fine   = 3;
    unsigned int n_components          = 1;
    unsigned int n_oversampling        = 1; // numbers::invalid_unsigned_int
    unsigned int n_subdivisions_coarse = 16;
    bool         LOD_stabilization     = true;
  };

  template <int dim>
  class LODProblem
  {
  public:
    LODProblem(const Parameters &params)
      : n_subdivisions_fine(params.n_subdivisions_fine)
      , n_components(params.n_components)
      , n_oversampling(params.n_oversampling)
      , n_subdivisions_coarse(params.n_subdivisions_coarse)
      , LOD_stabilization(params.LOD_stabilization)
      , comm(MPI_COMM_WORLD)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
      , repetitions(dim, n_subdivisions_coarse)
      , tria(comm,
             Triangulation<dim>::none,
             true,
             parallel::shared::Triangulation<dim>::partition_custom_signal)
    {
      AssertThrow(Utilities::MPI::n_mpi_processes(comm) <=
                    n_subdivisions_coarse,
                  ExcNotImplemented());
    }

    void
    make_grid()
    {
      Point<dim> p1;
      Point<dim> p2;

      for (unsigned int d = 0; d < dim; ++d)
        p2[d] = 1.0;

      const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
      const unsigned int stride =
        (repetitions[dim - 1] + n_procs - 1) / n_procs;

      tria.signals.create.connect([&, stride]() {
        for (const auto &cell : tria.active_cell_iterators())
          {
            unsigned int cell_index = cell->active_cell_index();

            for (unsigned int i = 0; i < dim - 1; ++i)
              cell_index /= repetitions[i];

            cell->set_subdomain_id(cell_index / stride);
          }
      });

      GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);
    }

    void
    setup_system()
    {
      // TODO
    }

    void
    setup_basis()
    {
      // TODO
    }

    void
    assemble_system()
    {
      // TODO
    }

    void
    solve()
    {
      TrilinosWrappers::SolverDirect solver;
      solver.solve(A_lod, solution_lod, rhs_lod);
    }

    void
    output_results()
    {
      // TODO
    }

    void
    run()
    {
      make_grid();
      setup_system();
      setup_basis();
      assemble_system();

      const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
      const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);
      const unsigned int stride =
        (repetitions[dim - 1] + n_procs - 1) / n_procs;
      unsigned int range_start =
        (my_rank == 0) ? 0 : ((stride * my_rank) * n_subdivisions_fine + 1);
      unsigned int range_end = stride * (my_rank + 1) * n_subdivisions_fine + 1;

      unsigned int face_dofs = n_components;
      for (unsigned int d = 0; d < dim - 1; ++d)
        face_dofs *= repetitions[d] * n_subdivisions_fine + 1;

      types::global_dof_index n_dofs_coarse = n_components;
      types::global_dof_index n_dofs_fine   = n_components;
      for (unsigned int d = 0; d < dim; ++d)
        {
          n_dofs_coarse *= repetitions[d];
          n_dofs_fine *= repetitions[d] * n_subdivisions_fine + 1;
        }

      AssertDimension(n_dofs_coarse, tria.n_active_cells() * n_components);

      IndexSet locally_owned_dofs_fine(n_dofs_fine);
      locally_owned_dofs_fine.add_range(
        face_dofs *
          std::min(range_start, repetitions[dim - 1] * n_subdivisions_fine + 1),
        face_dofs *
          std::min(range_end, repetitions[dim - 1] * n_subdivisions_fine + 1));

      Patch<dim> patch(n_subdivisions_fine, repetitions, n_components);

      IndexSet locally_owned_dofs_coarse(n_dofs_coarse);

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          for (unsigned int c = 0; c < n_components; ++c)
            locally_owned_dofs_coarse.add_index(
              cell->active_cell_index() * n_components + c);

      // 2) ininitialize sparsity pattern
      TrilinosWrappers::SparsityPattern sparsity_pattern_A_lod(
        locally_owned_dofs_coarse, comm);

      TrilinosWrappers::SparsityPattern sparsity_pattern_C(
        locally_owned_dofs_fine, comm);

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned()) // parallel for-loop
          {
            // A_lod sparsity pattern
            patch.reinit(cell, n_oversampling * 2);
            std::vector<types::global_dof_index> local_dof_indices_coarse;
            for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
              {
                const auto cell_index =
                  patch.create_cell_iterator(tria, cell)->active_cell_index();

                for (unsigned int c = 0; c < n_components; ++c)
                  local_dof_indices_coarse.emplace_back(
                    cell_index * n_components + c);
              }

            for (const auto &row_index : local_dof_indices_coarse)
              sparsity_pattern_A_lod.add_row_entries(row_index,
                                                     local_dof_indices_coarse);

            // C sparsity pattern
            patch.reinit(cell, n_oversampling);
            const auto                           n_dofs_patch = patch.n_dofs();
            std::vector<types::global_dof_index> local_dof_indices_fine(
              n_dofs_patch);
            patch.get_dof_indices(local_dof_indices_fine);

            AffineConstraints<double> patch_constraints;
            for (unsigned int d = 0; d < 2 * dim; ++d)
              patch.make_zero_boundary_constraints(d, patch_constraints);
            patch_constraints.close();

            for (unsigned int i = 0; i < n_dofs_patch; ++i)
              if (!patch_constraints.is_constrained(i))
                for (unsigned int c = 0; c < n_components; ++c)
                  sparsity_pattern_C.add_row_entries(
                    local_dof_indices_fine[i],
                    std::vector<types::global_dof_index>(
                      1, cell->active_cell_index() * n_components + c));
          }

      sparsity_pattern_A_lod.compress();
      sparsity_pattern_C.compress();

      // 3) initialize matrices
      A_lod.reinit(sparsity_pattern_A_lod);
      TrilinosWrappers::SparseMatrix C(sparsity_pattern_C);

      // 4) set dummy constraints
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned()) // parallel for-loop
          {
            patch.reinit(cell, n_oversampling);

            double H = 1.0 / n_subdivisions_coarse;
            double h = H / n_subdivisions_fine;

            const auto         n_dofs_patch  = patch.n_dofs();
            const unsigned int N_dofs_coarse = patch.n_cells();
            const unsigned int N_dofs_fine   = n_dofs_patch;
            std::vector<types::global_dof_index> local_dof_indices_fine(
              n_dofs_patch);
            patch.get_dof_indices(local_dof_indices_fine);

            AffineConstraints<double> patch_constraints;
            for (unsigned int d = 0; d < 2 * dim; ++d)
              if (patch.at_boundary(d))
                patch.make_zero_boundary_constraints(d, patch_constraints);
              else
                patch.make_zero_boundary_constraints(d, patch_constraints);
            patch_constraints.close();

            IndexSet patch_constraints_is(n_dofs_patch);
            for (const auto &l : patch_constraints.get_lines())
              patch_constraints_is.add_index(l.index);

            std::vector<Vector<double>> selected_basis_function(
              n_components, Vector<double>(n_dofs_patch));

            TrilinosWrappers::SparsityPattern sparsity_pattern(n_dofs_patch,
                                                               n_dofs_patch);
            patch.create_sparsity_pattern(patch_constraints, sparsity_pattern);
            sparsity_pattern.compress();

            TrilinosWrappers::SparseMatrix patch_stiffness_matrix(
              sparsity_pattern);
            FullMatrix<double> PT(N_dofs_fine, N_dofs_coarse);
            FullMatrix<double> P_Ainv_PT(N_dofs_coarse);
            FullMatrix<double> Ainv_PT(N_dofs_fine, N_dofs_coarse);
            // SLOD matrices
            std::vector<unsigned int>            internal_dofs_fine;
            std::vector<unsigned int>            all_dofs_fine; // to be filled
            std::vector<unsigned int> /*patch_*/ boundary_dofs_fine;
            std::vector<unsigned int>            domain_boundary_dofs_fine;

            patch.get_dofs_vectors(all_dofs_fine,
                                   internal_dofs_fine,
                                   /*patch_*/ boundary_dofs_fine,
                                   domain_boundary_dofs_fine);

            std::vector<unsigned int> all_dofs_coarse(all_dofs_fine.begin(),
                                                      all_dofs_fine.begin() +
                                                        N_dofs_coarse);

            unsigned int       considered_candidates = N_dofs_coarse - 1;
            const unsigned int N_boundary_dofs = boundary_dofs_fine.size();
            const unsigned int N_internal_dofs = internal_dofs_fine.size();

            FullMatrix<double> PT_boundary(N_boundary_dofs, N_dofs_coarse);
            FullMatrix<double> S_boundary(N_boundary_dofs, N_internal_dofs);

            Vector<double> PT_counter(N_dofs_fine);

            FESystem<dim>        fe(FE_Q_iso_Q1<dim>(n_subdivisions_fine),
                             n_components);
            const QIterated<dim> quadrature(QGauss<1>(2), n_subdivisions_fine);
            FEValues<dim>        fe_values(fe,
                                    quadrature,
                                    update_values | update_gradients |
                                      update_JxW_values);

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
              {
                Assert(i >= 1.0, ExcInternalError());
                i = 1.0 / i;
              }

            for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
              for (unsigned int i = 0; i < n_dofs_patch; ++i)
                PT[i][cell] *= PT_counter[i];

            if (LOD_stabilization && boundary_dofs_fine.size() > 0)
              {
                PT_boundary.extract_submatrix_from(PT,
                                                   boundary_dofs_fine,
                                                   all_dofs_coarse);
                if (true)
                  S_boundary.extract_submatrix_from(patch_stiffness_matrix,
                                                    boundary_dofs_fine,
                                                    internal_dofs_fine);
                else
                  {
                    for (unsigned int row_id = 0;
                         row_id < boundary_dofs_fine.size();
                         ++row_id)
                      for (unsigned int col_id = 0;
                           col_id < internal_dofs_fine.size();
                           ++col_id)
                        S_boundary.set(row_id,
                                       col_id,
                                       patch_stiffness_matrix.el(
                                         boundary_dofs_fine[row_id],
                                         internal_dofs_fine[col_id]));
                  }
              }

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

            auto central_cell_id = patch.cell_index(cell);

            if (!LOD_stabilization || (boundary_dofs_fine.size() == 0))
              // LOD
              // also in the case of : oversampling == 0 ||
              // or if the patch is the whole domain
              {
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    e_i[central_cell_id + n_components + c] = 1.0;
                    P_Ainv_PT.vmult(triple_product_inv_e_i, e_i);
                    Ainv_PT.vmult(selected_basis_function[c],
                                  triple_product_inv_e_i);
                  }
              }
            else // SLOD
              {
                FullMatrix<double>       BD(N_boundary_dofs, N_dofs_coarse);
                FullMatrix<double>       B_full(N_boundary_dofs, N_dofs_coarse);
                LAPACKFullMatrix<double> SVD(considered_candidates,
                                             considered_candidates);
                FullMatrix<double>       Ainv_PT_internal(N_internal_dofs,
                                                    N_dofs_coarse);

                Vector<double> internal_selected_basis_function(
                  N_internal_dofs);
                Vector<double> c_i(N_internal_dofs);
                internal_selected_basis_function = 0.0;

                for (unsigned int c = 0; c < n_components; ++c)
                  selected_basis_function[c] = 0.0;

                Ainv_PT_internal.extract_submatrix_from(Ainv_PT,
                                                        internal_dofs_fine,
                                                        all_dofs_coarse);
                S_boundary.mmult(B_full, Ainv_PT_internal);

                // creating the matrix B_full using all components from all
                // candidates
                PT_boundary *= -1;
                B_full.mmult(BD, P_Ainv_PT);
                PT_boundary.mmult(BD, P_Ainv_PT, true);

                for (unsigned int d = 0; d < n_components; ++d)
                  {
                    Vector<double> B_d0(N_boundary_dofs);

                    for (unsigned int i = 0; i < N_boundary_dofs; ++i)
                      B_d0[i] = BD(i, central_cell_id * n_components + d);

                    Vector<double> d_i(considered_candidates);
                    Vector<double> BDTBD0(considered_candidates);
                    d_i    = 0;
                    BDTBD0 = 0;

                    // std::vector<unsigned int> other_phi(all_dofs_fine.begin()
                    // + 1,
                    //                                     all_dofs_fine.begin()
                    //                                     +
                    //                                       N_dofs_coarse);
                    std::vector<unsigned int> other_phi(all_dofs_fine.begin(),
                                                        all_dofs_fine.begin() +
                                                          N_dofs_coarse);
                    other_phi.erase(other_phi.begin() +
                                    central_cell_id * n_components + d);

                    {
                      FullMatrix<double> newBD(N_boundary_dofs,
                                               considered_candidates);
                      FullMatrix<double> BDTBD(considered_candidates,
                                               considered_candidates);

                      Assert(
                        other_phi.size() == considered_candidates,
                        ExcNotImplemented(
                          "inconsistent number of candidates basis function on the patch"));
                      std::vector<unsigned int> boundary_dofs_vector_temp(
                        all_dofs_fine.begin(),
                        all_dofs_fine.begin() + N_boundary_dofs);

                      newBD.extract_submatrix_from(BD,
                                                   boundary_dofs_vector_temp,
                                                   other_phi);

                      newBD.Tmmult(BDTBD, newBD);

                      newBD.Tvmult(BDTBD0, B_d0);

                      SVD.copy_from(BDTBD);
                    }

                    SVD.compute_inverse_svd(1e-15); // stores U V as normal, but
                                                    // 1/singular_value_i
                    d_i = 0.0;
                    SVD.vmult(d_i, BDTBD0);
                    d_i *= -1;
                    auto U  = SVD.get_svd_u();
                    auto Vt = SVD.get_svd_vt();

                    AssertDimension(SVD.m(), SVD.n());
                    AssertDimension(U.m(), U.n());
                    AssertDimension(Vt.m(), Vt.n());
                    AssertDimension(U.m(), Vt.n());
                    AssertDimension(U.m(), SVD.n());
                    AssertDimension(U.m(), considered_candidates);

                    for (int i = (considered_candidates - 1); i >= 0; --i)
                      {
                        if (d_i.linfty_norm() < 0.5)
                          break;
                        Vector<double> uT(considered_candidates);
                        Vector<double> v(considered_candidates);
                        // for (auto j : all_dofs_coarse)
                        for (unsigned int j = 0; j < considered_candidates; ++j)
                          {
                            uT[j] = U(j, i);
                            v[j]  = Vt(i, j);
                          }
                        FullMatrix<double> vuT(considered_candidates,
                                               considered_candidates);
                        // do uT scalar BDTBD0 first
                        vuT.outer_product(v, uT);
                        Vector<double> correction(d_i.size());
                        vuT.vmult(correction, BDTBD0);
                        correction *= // Sigma_minus1(i, i); //
                          SVD.singular_value(i);

                        d_i += correction;
                      }

                    Vector<double> DeT(N_dofs_coarse);
                    e_i                                     = 0.0;
                    e_i[central_cell_id * n_components + d] = 1.0;
                    P_Ainv_PT.vmult(DeT, e_i);
                    c_i = DeT;

                    // for (unsigned int index = 0; index <
                    // considered_candidates;
                    // ++index)
                    for (unsigned int index = 0; index < other_phi.size();
                         ++index)
                      {
                        e_i                   = 0.0;
                        e_i[other_phi[index]] = 1.0;

                        P_Ainv_PT.vmult(DeT, e_i);

                        DeT *= d_i[index];

                        c_i += DeT;
                      }

                    Ainv_PT_internal.vmult(internal_selected_basis_function,
                                           c_i);
                    unsigned int N_boundary_dofs = 4 * n_subdivisions_fine;
                    // somehow the following does not work
                    // internal_selected_basis_function.extract_subvector_to(internal_selected_basis_function.begin(),
                    // internal_selected_basis_function.end(),
                    // selected_basis_function.begin()+N_boundary_dofs);
                    for (unsigned int id = 0;
                         id <
                         (selected_basis_function[d].size() - N_boundary_dofs);
                         ++id)
                      selected_basis_function[d][id + N_boundary_dofs] =
                        internal_selected_basis_function[id];
                  }
              }

            for (unsigned int c = 0; c < n_components; ++c)
              {
                selected_basis_function[c] /=
                  selected_basis_function[c].l2_norm();

                patch_constraints.set_zero(selected_basis_function[c]);

                for (unsigned int i = 0; i < n_dofs_patch; ++i)
                  if (selected_basis_function[c][i] != 0.0)
                    C.set(local_dof_indices_fine[i],
                          cell->active_cell_index(),
                          selected_basis_function[c][i]);
              }
          }

      C.compress(VectorOperation::values::insert);

      // 5) convert sparse matrix C to shifted AffineConstraints
      IndexSet constraints_lod_fem_locally_owned_dofs(n_dofs_fine +
                                                      n_dofs_coarse);
      constraints_lod_fem_locally_owned_dofs.add_indices(
        locally_owned_dofs_coarse);
      constraints_lod_fem_locally_owned_dofs.add_indices(
        locally_owned_dofs_fine, n_dofs_coarse);

      IndexSet constraints_lod_fem_locally_stored_constraints =
        constraints_lod_fem_locally_owned_dofs;

      IndexSet locally_relevant_cells = locally_owned_dofs_coarse;

      for (const auto row : locally_owned_dofs_fine) // parallel for-loop
        {
          for (auto entry = C.begin(row); entry != C.end(row); ++entry)
            {
              constraints_lod_fem_locally_stored_constraints.add_index(
                entry->column()); // coarse
            }
        }

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned()) // parallel for-loop
          {
            patch.reinit(cell, n_oversampling);

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
      for (const auto row : locally_owned_dofs_fine) // parallel for-loop
        {
          std::vector<std::pair<types::global_dof_index, double>> dependencies;

          for (auto entry = C.begin(row); entry != C.end(row); ++entry)
            dependencies.emplace_back(entry->column(), entry->value());

          if (true || !dependencies.empty())
            constraints_lod_fem.add_constraint(row + n_dofs_coarse,
                                               dependencies);
        }

      constraints_lod_fem.make_consistent_in_parallel(
        constraints_lod_fem_locally_owned_dofs,
        constraints_lod_fem_locally_stored_constraints,
        comm);
      constraints_lod_fem.close();

      for (const auto row :
           constraints_lod_fem_locally_stored_constraints) // parallel for-loop
        if (const auto constraint_entries =
              constraints_lod_fem.get_constraint_entries(row))
          {
            for (const auto &[j, _] : *constraint_entries)
              locally_relevant_cells.add_index(j);
          }

      rhs_lod.reinit(locally_owned_dofs_coarse, locally_relevant_cells, comm);
      solution_lod.reinit(rhs_lod);

      // 6) assembly LOD matrix
      FESystem<dim> fe(FE_Q_iso_Q1<dim>(n_subdivisions_fine), n_components);
      const QIterated<dim> quadrature(QGauss<1>(2), n_subdivisions_fine);
      FEValues<dim>        fe_values(
        fe, quadrature, update_values | update_gradients | update_JxW_values);

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned()) // parallel for-loop
          {
            Patch<dim> patch(n_subdivisions_fine, repetitions, n_components);
            patch.reinit(cell, 0);

            fe_values.reinit(cell);

            const unsigned int n_dofs_per_cell = patch.n_dofs();

            // a) compute FEM element stiffness matrix
            FullMatrix<double> cell_matrix_fem(n_dofs_per_cell,
                                               n_dofs_per_cell);
            Vector<double>     cell_rhs_fem(n_dofs_per_cell);

            for (const unsigned int q_index :
                 fe_values.quadrature_point_indices())
              {
                for (const unsigned int i : fe_values.dof_indices())
                  for (const unsigned int j : fe_values.dof_indices())
                    cell_matrix_fem(i, j) += (fe_values.shape_grad(i, q_index) *
                                              fe_values.shape_grad(j, q_index) *
                                              fe_values.JxW(q_index));

                for (const unsigned int i : fe_values.dof_indices())
                  cell_rhs_fem(i) += (fe_values.shape_value(i, q_index) * 1. *
                                      fe_values.JxW(q_index));
              }

            // b) assemble into LOD matrix by using constraints
            std::vector<types::global_dof_index> local_dof_indices(
              n_dofs_per_cell);
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

      // 7) solve LOD system
      this->solve();
      this->output_results();

      // 8) convert to FEM solution
      LinearAlgebra::distributed::Vector<double> solution_fem(
        locally_owned_dofs_fine, comm);

      solution_lod.update_ghost_values();
      for (const auto i : locally_owned_dofs_fine)
        if (const auto constraint_entries =
              constraints_lod_fem.get_constraint_entries(i + n_dofs_coarse))
          {
            double new_value = 0.0;
            for (const auto &[j, weight] : *constraint_entries)
              new_value += weight * solution_lod[j];

            solution_fem[i] = new_value;
          }

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

      pcout << solution_lod.l2_norm() << std::endl;
      pcout << solution_fem.l2_norm() << std::endl;

      Vector<double> ranks(tria.n_active_cells());
      ranks = my_rank;
      data_out.add_data_vector(ranks, "rank");

      data_out.build_patches(mapping, n_subdivisions_fine);

      const std::string file_name = "solution.vtu";

      data_out.write_vtu_in_parallel(file_name, comm);
    }

  private:
    const unsigned int n_subdivisions_fine;
    const unsigned int n_components;
    const unsigned int n_oversampling;
    const unsigned int n_subdivisions_coarse;
    const bool         LOD_stabilization;

    MPI_Comm           comm;
    ConditionalOStream pcout;

    std::vector<unsigned int> repetitions;

    parallel::shared::Triangulation<dim> tria;

    TrilinosWrappers::SparseMatrix A_lod;

    LinearAlgebra::distributed::Vector<double> rhs_lod;
    LinearAlgebra::distributed::Vector<double> solution_lod;
  };

} // namespace Step96

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  Step96::Parameters params;

  Step96::LODProblem<2> problem(params);
  problem.run();
}