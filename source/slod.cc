#include <slod.h>

template <int dim>
SLOD<dim>::SLOD(const SLODParameters<dim, dim> &par)
  : par(par)
  , mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , tria(mpi_communicator)
  , dof_handler(tria)
{}

template <int dim>
void
SLOD<dim>::print_parameters() const
{
  pcout << "Running SLODProblem" << std::endl;
  par.prm.print_parameters(par.output_directory + "/" + "used_parameters_" +
                             std::to_string(dim) + ".prm",
                           ParameterHandler::Short);
}

template <int dim>
void
SLOD<dim>::make_fe()
{
  TimerOutput::Scope t(computing_timer, "make FE spaces");
  // fe_coarse = std::make_unique<FESystem<dim>>(FE_DGQ<dim>(0), 1);
  fe_coarse = std::make_unique<FE_DGQ<dim>>(FE_DGQ<dim>(0));
  dof_handler.distribute_dofs(*fe_coarse);

  auto locally_owned_dofs = dof_handler.locally_owned_dofs();
  auto locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  // TODO: Parameter adaptable parameters
  VectorTools::interpolate_boundary_values(
    // dof_handler, 0, par.bc, constraints);
    dof_handler,
    0,
    Functions::ConstantFunction<dim, double>(0),
    constraints);
  constraints.close();

  DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler,
                                  sparsity_pattern,
                                  constraints,
                                  /*keep constrained dofs*/ false);
  SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);
  global_stiffness_matrix.reinit(locally_owned_dofs,
                                 locally_owned_dofs,
                                 sparsity_pattern,
                                 mpi_communicator);

  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  fe_fine =
    std::make_unique<FE_Q_iso_Q1<dim>>(FE_Q_iso_Q1<dim>(par.n_subdivisions));
  // std::make_unique<FESystem<dim>>(FE_Q_iso_Q1<dim>(par.n_subdivisions), 1);
  // TODO: set order
  quadrature_fine = std::make_unique<Quadrature<dim>>(
    QIterated<dim>(QGauss<1>(2), par.n_subdivisions));
}

template <int dim>
void
SLOD<dim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "create grid");
  GridGenerator::hyper_cube(tria);
  tria.refine_global(par.n_global_refinements);
}

template <int dim>
void
SLOD<dim>::create_patches()
{
  TimerOutput::Scope t(computing_timer, "create patches");

  locally_owned_patches =
    Utilities::MPI::create_evenly_distributed_partitioning(
      mpi_communicator, tria.n_active_cells());
  global_to_local_cell_map.resize(tria.n_global_active_cells());
  // patches = TrilinosWrappers::MPI::Vector(locally_owned_patches,
  //                                          mpi_communicator);
  // patches = 0;
  // Queue for patches for which neighbours should be added
  std::vector<typename DoFHandler<dim>::active_cell_iterator> patch_iterators;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_index = cell->active_cell_index();
      // if (locally_owned_patches.is_element(cell_index))
      // previous line allows all processors to create the same patches but then
      // only work on the locally_owned
      {
        // for each cell we create its patch and add it to the global vector
        // of patches
        auto patch = &patches.emplace_back();
        // AssertIndexRange(cell_index, patches.size());
        // auto patch = &patches[cell_index];
        patch_iterators.clear();
        patch_iterators.push_back(cell);
        // The iterators for level l are in the range [l_start, l_end) of
        // patch_iterators
        unsigned int l_start = 0;
        unsigned int l_end   = 1;
        patch->cells.push_back(cell);
        patch->cell_indices.set_size(tria.n_active_cells());
        patch->cell_indices.add_index(cell_index);
        for (unsigned int l = 1; l <= par.oversampling; l++)
          {
            for (unsigned int i = l_start; i <= l_end; i++)
              {
                // AssertIndexRange(i, patch_iterators.size());
                for (auto vertex : patch_iterators[i]->vertex_indices())
                  {
                    for (const auto &neighbour :
                         GridTools::find_cells_adjacent_to_vertex(dof_handler,
                                                                  vertex))
                      {
                        if (!patch->cell_indices.is_element(
                              neighbour->active_cell_index()))
                          {
                            patch_iterators.push_back(neighbour);
                            // CHECK: DOES NOT WORK FOR oversapling = 2
                            patch->cells.push_back(neighbour);
                          }
                        patch->cell_indices.add_index(
                          neighbour->active_cell_index());
                      }
                  }
              }
            l_start = l_end;
            l_end   = patch_iterators.size();
          }
      }
    }
  // patches.compress(VectorOperation::add);
  // MPI::Barrier();
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      pcout << "   Number of coarse cell = " << tria.n_active_cells()
            << ", number of patches = " << patches.size()
            << " (locally owned: " << locally_owned_patches.n_elements() << ")"
            << std::endl;
    }

  // For patches at the border, find the neighbouring patch that is not at the
  // border and which completely contains this patch
  //   for (auto &patch : patches) {
  //     if (patch.num_basis_vectors == 0) {
  //       bool success = false;
  //       for (auto i : patch.cells)
  //       {
  //         Patch neighbour = patches[i];
  //         if (neighbour.num_basis_vectors == 0) continue;
  //         // Check that the patch around i contains all of the cells
  //         // of the patch specified by pair
  //         bool contains_patch = true;
  //         for (auto j : patch.cells) {
  //           if (!neighbour.cells.is_element(j)) {
  //             contains_patch = false;
  //             break;
  //           }
  //         }
  //         if (contains_patch) {
  //           neighbour.num_basis_vectors++;
  //           success = true;
  //           break;
  //         }
  //       }
  //       Assert(success, ExcNotImplemented());
  //     }
  //   }
}

template <int dim>
void
SLOD<dim>::check_nested_patches()
{
  std::cout << "   checking nested patches";
  TimerOutput::Scope t(computing_timer, "check nested patches");

  for (auto current_patch_id : locally_owned_patches)
    {
      AssertIndexRange(current_patch_id, patches.size());
      auto       current_patch    = &patches[current_patch_id];
      const auto current_cell_set = current_patch->cell_indices;
      for (auto cell_to_check : current_cell_set)
        {
          if (!(cell_to_check == current_patch_id))
            {
              AssertIndexRange(cell_to_check, patches.size());
              auto set_to_check = patches[cell_to_check].cell_indices;
              set_to_check.subtract_set(current_cell_set);
              if (set_to_check.is_empty())
                current_patch->contained_patches++;
            }
        }
    }
  std::cout << ": done" << std::endl;
}


template <int dim>
void
SLOD<dim>::output_results() const
{
  TimerOutput::Scope t(computing_timer, "Output results");

  // std::vector<std::string> solution_names(dim, "coarse_solution");
  std::string solution_names("coarse_solution");
  // std::vector<std::string> exact_solution_names(dim, "exact_solution");

  // auto exact_vec(solution);
  // VectorTools::interpolate(dof_handler, par.exact_solution, exact_vec);
  // to be added for MPI
  // auto exact_vec_locally_relevant(locally_relevant_solution.block(0));
  // exact_vec_locally_relevant = exact_vec;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      //     dim, DataComponentInterpretation::component_is_part_of_vector);
      dim,
      DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  // data_out.add_data_vector(solution, solution_names);
  data_out.add_data_vector(solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  // data_out.add_data_vector(exact_vec,
  //                          exact_solution_names,
  //                          DataOut<dim>::type_dof_data,
  //                          data_component_interpretation);
  Vector<float> subdomain(tria.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");
  data_out.build_patches();
  const std::string filename = par.output_name + ".vtu";
  data_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                 mpi_communicator);

  // std::ofstream pvd_solutions(par.output_directory + "/" + par.output_name +
  //                                 ".pvd");
}
/*
template <int dim>
LinearOperator<LinearAlgebra::distributed::Vector<double>>
transfer_operator(
  MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<double>> &transfer,
  unsigned int                                                         m,
  unsigned int                                                         n)
{
  LinearOperator<LinearAlgebra::distributed::Vector<double>> return_op{
    internal::LinearOperatorImplementation::EmptyPayload()};

  return_op.vmult =
    [&transfer](LinearAlgebra::distributed::Vector<double> &      out,
                const LinearAlgebra::distributed::Vector<double> &in) {
      out = 0;
      transfer.prolongate_and_add(out, in);
    };
  return_op.vmult_add =
    [&transfer](LinearAlgebra::distributed::Vector<double> &      out,
                const LinearAlgebra::distributed::Vector<double> &in) {
      transfer.prolongate_and_add(out, in);
    };
  return_op.Tvmult =
    [&transfer](LinearAlgebra::distributed::Vector<double> &      out,
                const LinearAlgebra::distributed::Vector<double> &in) {
      out = 0;
      transfer.restrict_and_add(out, in);
    };
  return_op.Tvmult_add =
    [&transfer](LinearAlgebra::distributed::Vector<double> &      out,
                const LinearAlgebra::distributed::Vector<double> &in) {
      transfer.restrict_and_add(out, in);
    };
  return_op.reinit_domain_vector =
    [&m](LinearAlgebra::distributed::Vector<double> &out,
         bool omit_zeroing_entries) { out.reinit(m, omit_zeroing_entries); };
  return_op.reinit_range_vector =
    [&n](LinearAlgebra::distributed::Vector<double> &out,
         bool omit_zeroing_entries) { out.reinit(n, omit_zeroing_entries); };

  return return_op;
}
*/
const unsigned int SPECIAL_NUMBER = 69;

template <int dim>
void
SLOD<dim>::compute_basis_function_candidates_using_SVD()
{
  /*
  DoFHandler<dim> dh_coarse_patch;

  // TODO: reinit in loop
  LA::MPI::SparseMatrix     patch_stiffness_matrix;
  AffineConstraints<double> internal_boundary_constraints;

  IndexSet local_dofs_fine;
  IndexSet local_dofs_coarse;

  MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<double>> transfer;

  std::vector<std::complex<double>> eigenvalues;

  std::vector<LinearAlgebra::distributed::Vector<double>> right_hand_sides;
  for (unsigned int j = 0; j < par.num_basis_vectors; j++)
    {
      right_hand_sides.emplace_back();
    }

  SolverControl solver_control(100, 1e-9);
  ArpackSolver  eig_solver(solver_control);

  const auto locally_owned_patches =
    Utilities::MPI::create_evenly_distributed_partitioning(mpi_communicator,
                                                           patches.size());
  // for (unsigned int patch_id = 0; patch_id < patches.size(); patch_id++) {
  for (auto current_patch_id : locally_owned_patches)
    {
      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];
      //   if (patches[i].num_basis_vectors == 0) continue;
      internal_boundary_constraints.clear();
      create_mesh_for_patch(*current_patch, internal_boundary_constraints);
      current_patch->dh_fine->reinit(current_patch->sub_tria);
      current_patch->dh_fine->distribute_dofs(*fe_fine);
      dh_coarse_patch.distribute_dofs(*fe_coarse);

      // TODO: Can fe_fine be used for different patches at the same time?
      assemble_stiffness_for_patch(*current_patch, patch_stiffness_matrix);
      // VectorTools::interpolate_boundary_values(
      //   *current_patch->dh_fine,
      //   SPECIAL_NUMBER,
      //   Functions::ConstantFunction<dim, double>(0),
      //   internal_boundary_constraints);

      const auto A =
        linear_operator<LinearAlgebra::distributed::Vector<double>>(
          patch_stiffness_matrix);
      const auto A0 =
        constrained_linear_operator<LinearAlgebra::distributed::Vector<double>>(
          internal_boundary_constraints, A);

      internal_boundary_constraints.close();
      // Following the method of contrained_linear_operator, construct C^T * A *
      // Id_c
      const auto C = distribute_constraints_linear_operator<
        LinearAlgebra::distributed::Vector<double>>(
        internal_boundary_constraints, A);
      const auto Id_c = project_to_constrained_linear_operator<
        LinearAlgebra::distributed::Vector<double>>(
        internal_boundary_constraints, A);
      const auto A1 = transpose_operator(C) * A * Id_c;

      auto A0_inv = A0;
      // TODO: preconditioning
      // TODO: solver contro
      SolverCG<LinearAlgebra::distributed::Vector<double>> cg_A(solver_control);
      A0_inv = inverse_operator(A0, cg_A);

      local_dofs_fine.clear();
      local_dofs_fine.set_size(current_patch->dh_fine->n_dofs());
      local_dofs_fine.add_range(0, current_patch->dh_fine->n_dofs());
      local_dofs_coarse.clear();
      local_dofs_fine.set_size(dh_coarse_patch.n_dofs());
      local_dofs_coarse.add_range(0, dh_coarse_patch.n_dofs());

      transfer.reinit_polynomial_transfer(dh_coarse_patch,
                                          *current_patch->dh_fine,
                                          AffineConstraints(),
                                          AffineConstraints());
      const auto P = transfer_operator(transfer,
                                       dh_coarse_patch.n_dofs(),
                                       current_patch->dh_fine->n_dofs());

      SolverCG<LinearAlgebra::distributed::Vector<double>> cg_X(solver_control);
      const LinearOperator<LinearAlgebra::distributed::Vector<double>> X =
        P * A0_inv * A1;
      const auto X_sq     = transpose_operator(X) * X;
      const auto X_sq_inv = inverse_operator(X_sq, cg_X);

      for (unsigned int j = 0; j < par.num_basis_vectors; j++)
        {
          right_hand_sides[j].reinit(dh_coarse_patch.n_dofs());
        }
      eig_solver.solve(X_sq,
                       identity_operator(X_sq),
                       X_sq_inv,
                       eigenvalues,
                       right_hand_sides,
                       par.num_basis_vectors);

      for (unsigned int j = 0; j < par.num_basis_vectors; j++)
        {
          LinearAlgebra::distributed::Vector<double> right_hand_side_fine;
          right_hand_side_fine = transpose_operator(P) * right_hand_sides[j];
          right_hand_side_fine = A0_inv * right_hand_side_fine;
          current_patch->basis_function_candidates.push_back(
            right_hand_side_fine);
        }
    }
    */
}

template <int dim>
void
SLOD<dim>::compute_basis_function_candidates()
{
  std::cout << "   compute basis function candidate";
  DoFHandler<dim> dh_coarse_patch;
  DoFHandler<dim> dh_fine_patch;

  // using VectorType = LinearAlgebra::distributed::Vector<double>;
  // would be nice to have LA::distributed
  using VectorType = Vector<double>;

  // TODO: reinit in loop
  LA::MPI::SparseMatrix     patch_stiffness_matrix;
  AffineConstraints<double> internal_boundary_constraints;

  IndexSet local_dofs_fine;
  IndexSet local_dofs_coarse;

  for (auto current_patch_id : locally_owned_patches)
    {
      //  MGTwoLevelTransfer<dim, VectorType> transfer;

      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];

      create_mesh_for_patch(*current_patch);
      dh_fine_patch.reinit(current_patch->sub_tria);
      dh_fine_patch.distribute_dofs(*fe_fine);
      // current_patch->dh_fine =
      // std::make_unique<DoFHandler<dim>>(dh_fine_patch);
      dh_coarse_patch.reinit(current_patch->sub_tria);
      dh_coarse_patch.distribute_dofs(*fe_coarse);

      std::vector<IndexSet> relevant_dofs;
      relevant_dofs.resize(1);
      DoFTools::extract_locally_relevant_dofs(dh_fine_patch, relevant_dofs[0]);

      {
        // Homogeneus Dirichlet at the boundary of the patch, identifies by
        // specialnumber
        internal_boundary_constraints.clear();
        internal_boundary_constraints.reinit(relevant_dofs[0]);
        // !!!!! is it ok to pass dh_fine_patch as copy? isn't it highly stupid?
        VectorTools::interpolate_boundary_values(
          dh_fine_patch,
          SPECIAL_NUMBER,
          Functions::ConstantFunction<dim, double>(0),
          internal_boundary_constraints);
        internal_boundary_constraints.close();
      }
      {
        DynamicSparsityPattern dsp(relevant_dofs[0]);
        auto                   owned_dofs = relevant_dofs;
        DoFTools::make_sparsity_pattern(dh_fine_patch,
                                        dsp,
                                        internal_boundary_constraints,
                                        false);
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   owned_dofs[0],
                                                   mpi_communicator,
                                                   relevant_dofs[0]);
        patch_stiffness_matrix.clear();
        patch_stiffness_matrix.reinit(owned_dofs[0],
                                      owned_dofs[0],
                                      dsp,
                                      mpi_communicator);
      }

      assemble_stiffness_for_patch( //*current_patch,
        patch_stiffness_matrix,
        dh_fine_patch);

      // do we actually need A?
      // i think A with the constrained ( see make sparsity pattern) might be
      // already what we need as A0
      const auto A  = linear_operator<VectorType>(patch_stiffness_matrix);
      const auto A0 = // S
        constrained_linear_operator<VectorType>(internal_boundary_constraints,
                                                A);
      auto A0_inv = A0;

      SolverCG<VectorType> cg_A(par.fine_solver_control);
      A0_inv = inverse_operator(A0, cg_A);

      local_dofs_fine.clear();
      local_dofs_fine.set_size(dh_fine_patch.n_dofs());
      local_dofs_fine.add_range(0, dh_fine_patch.n_dofs());
      local_dofs_coarse.clear();
      local_dofs_coarse.set_size(dh_coarse_patch.n_dofs());
      local_dofs_coarse.add_range(0, dh_coarse_patch.n_dofs());

      //  FullMatrix<double> S_inv_Pt(Nfine, Ncoarse);
      // const auto S_inv = linear_operator<LA::MPI::Vector>(A0_inv);
      //  S_inv_Pt = A0_inv* projection_matrix;
      // const auto P = linear_operator<LA::MPI::Vector>(projection_matrix);
      // const auto Pt = transpose_operator(P);
      //  const auto S_inv_Pt = A0_inv * Pt;
      //  auto P_bar_S_inv_Pt_inv =
      //  linear_operator<LA::MPI::Vector>(projection_matrix) * S_inv_Pt;
      //  P_bar_S_inv_Pt_inv.invert();
      //  auto c_i = P_bar_S_inv_Pt_inv * e_i;
      //  c_i = S_inv_Pt * c_i;

      // create projection matrix from fine to coarse cell (DG)
      FullMatrix<double> projection_matrix(fe_coarse->n_dofs_per_cell(),
                                           fe_fine->n_dofs_per_cell());
      FETools::get_projection_matrix(*fe_fine, *fe_coarse, projection_matrix);

      // averaging (inverse of P0 mass matrix)
      VectorType valence_coarse(dh_coarse_patch.n_dofs());
      VectorType local_identity_coarse(fe_coarse->n_dofs_per_cell());
      local_identity_coarse = 1.0;

      for (const auto &cell : dh_coarse_patch.active_cell_iterators())
        cell->distribute_local_to_global(local_identity_coarse, valence_coarse);

      for (auto &elem : valence_coarse)
        elem = 1.0 / elem;

      // define interapolation function and its transposed
      const auto projectT = [&](auto &dst, const auto &src) {
        VectorType vec_local_coarse(fe_coarse->n_dofs_per_cell());
        VectorType vec_local_fine(fe_fine->n_dofs_per_cell());
        VectorType weights(fe_coarse->n_dofs_per_cell());

        for (const auto &cell : current_patch->sub_tria.active_cell_iterators())
          // should be locally owned
          {
            const auto cell_coarse =
              cell->as_dof_handler_iterator(dh_coarse_patch);
            const auto cell_fine = cell->as_dof_handler_iterator(dh_fine_patch);

            cell_fine->get_dof_values(src, vec_local_fine);

            projection_matrix.vmult(vec_local_coarse, vec_local_fine);

            cell_coarse->get_dof_values(valence_coarse, weights);
            vec_local_coarse.scale(weights);

            cell_coarse->distribute_local_to_global(vec_local_coarse, dst);
          }
      };

      // const auto project = [&](auto &dst, const auto &src) {
      //   VectorType vec_local_coarse(fe_coarse->n_dofs_per_cell());
      //   VectorType vec_local_fine(fe_fine->n_dofs_per_cell());
      //   VectorType weights(fe_coarse->n_dofs_per_cell());

      //   for (const auto &cell : current_patch->sub_tria.active_cell_iterators())
      //     // should be locally_owned
      //     {
      //       const auto cell_coarse =
      //         cell->as_dof_handler_iterator(dh_coarse_patch);
      //       const auto cell_fine = cell->as_dof_handler_iterator(dh_fine_patch);

      //       cell_coarse->get_dof_values(src, vec_local_coarse);

      //       cell_coarse->get_dof_values(valence_coarse, weights);
      //       vec_local_coarse.scale(weights);

      //       projection_matrix.Tvmult(vec_local_fine, vec_local_coarse);

      //       cell_fine->distribute_local_to_global(vec_local_fine, dst);
      //     }
      // };

      // Specialization of projection for the case where src is the P0 basis function of a single cell
      // Works only for P0 coarse elements
      const auto project_cell = [&](auto &dst, const auto &cell) {
        Assert(fe_coarse->n_dofs_per_cell() == 1, "Only works with DGQ1 elements");
        VectorType vec_local_coarse(fe_coarse->n_dofs_per_cell());
        VectorType vec_local_fine(fe_fine->n_dofs_per_cell());
        VectorType weights(fe_coarse->n_dofs_per_cell());

          const auto cell_coarse =
            cell->as_dof_handler_iterator(dh_coarse_patch);
          const auto cell_fine = cell->as_dof_handler_iterator(dh_fine_patch);

          // cell_coarse->get_dof_values(src, vec_local_coarse);
          vec_local_coarse[0] = 1.0;

          cell_coarse->get_dof_values(valence_coarse, weights);
          vec_local_coarse.scale(weights);

          projection_matrix.Tvmult(vec_local_fine, vec_local_coarse);

          cell_fine->distribute_local_to_global(vec_local_fine, dst);
      };

      VectorType temp(dh_fine_patch.n_dofs());
      VectorType temp1(dh_coarse_patch.n_dofs());
      VectorType u_i(dh_fine_patch.n_dofs());
      VectorType v_i(dh_coarse_patch.n_dofs());
      FullMatrix<double> triple_product(dh_coarse_patch.n_dofs());
      FullMatrix<double> A_inv_P(dh_fine_patch.n_dofs(), dh_coarse_patch.n_dofs());

      for (const auto &cell : current_patch->sub_tria.active_cell_iterators())
        {
          unsigned int i = cell->active_cell_index();
          temp = 0.0;
          u_i = 0.0;

          project_cell(temp, cell);
          u_i = A0_inv * temp;
          v_i     = 0.0;
          projectT(v_i, u_i);
          for (unsigned int j = 0; j < dh_coarse_patch.n_dofs(); j++) {
            triple_product(j, i) = v_i[j];
            // std::cout << v_i[j] << "\t";
          }
          // std::cout << std::endl;
          // std::cout << std::endl;
          for (unsigned int j = 0; j < dh_fine_patch.n_dofs(); j++) {
            A_inv_P(j, i) = u_i[j];
            // std::cout << u_i[j] << "\t";
          }
          // std::cout << std::endl;
        }
      triple_product.gauss_jordan();
      {
        v_i     = 0.0;
        // 0 is the index of the central cell
        // (this is also the central dof because we use P0 elements)
        v_i[0] = 1.0;
        triple_product.vmult(temp1, v_i);
        A_inv_P.vmult(u_i, temp1);
        current_patch->basis_function_candidates.push_back(u_i);
        temp1 = A0 * u_i;
        current_patch->basis_function_candidates_premultiplied.push_back(temp1);
      }


      dh_fine_patch.clear();
    }
  std::cout << ": done" << std::endl;
}

template <int dim>
void
SLOD<dim>::create_mesh_for_patch(Patch<dim> &current_patch)
{
  current_patch.sub_tria.clear();

  // copy manifolds
  for (const auto i : tria.get_manifold_ids())
    if (i != numbers::flat_manifold_id)
      current_patch.sub_tria.set_manifold(i, tria.get_manifold(i));

  // renumerate vertices
  std::vector<unsigned int> new_vertex_indices(tria.n_vertices(), 0);

  for (const auto &cell : current_patch.cells)
    for (const unsigned int v : cell->vertex_indices())
      new_vertex_indices[cell->vertex_index(v)] = 1;

  for (unsigned int i = 0, c = 0; i < new_vertex_indices.size(); ++i)
    if (new_vertex_indices[i] == 0)
      new_vertex_indices[i] = numbers::invalid_unsigned_int;
    else
      new_vertex_indices[i] = c++;

  // collect points
  std::vector<Point<dim>> sub_points;
  for (unsigned int i = 0; i < new_vertex_indices.size(); ++i)
    if (new_vertex_indices[i] != numbers::invalid_unsigned_int)
      sub_points.emplace_back(tria.get_vertices()[i]);

  // create new cell and data
  std::vector<CellData<dim>> coarse_cells_of_patch;

  for (const auto &cell : current_patch.cells)
    {
      CellData<dim> new_cell(cell->n_vertices());

      for (const auto v : cell->vertex_indices())
        new_cell.vertices[v] = new_vertex_indices[cell->vertex_index(v)];

      new_cell.material_id = cell->material_id();
      new_cell.manifold_id = cell->manifold_id();

      coarse_cells_of_patch.emplace_back(new_cell);
    }

  // create coarse mesh on the patch
  current_patch.sub_tria.create_triangulation(sub_points,
                                              coarse_cells_of_patch,
                                              {});

  /*

    //now we need to save into internal_boundary_constraints the info of who's
    on the border

    std::vector<unsigned int> local_face_dof_indices (fe_fine->dofs_per_line);
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    auto coarse_cell_of_patch = current_patch.sub_tria.begin(0);

    for (auto coarse_cell : current_patch.cells) // not sure if the indeces are
    the same
    {
      for (const auto f : coarse_cell->face_indices())
      {
        auto active_sub_cells =
    coarse_cell_of_patch->face(f)->n_active_descendants();
        // if coarse_face is at boundary then also the fine_fce is at the
    boundary and the global constrain will prevail if
    (!coarse_cell->face(f)->at_boundary() &&
    coarse_cell_of_patch->face(f)->at_boundary())
          {
            for (unsigned int child_idx = 0; child_idx < active_sub_cells;
    child_idx ++)
            {
              auto sub_cell = coarse_cell_of_patch->face(f)->child(child_idx);
                sub_cell->get_dof_indices(local_face_dof_indices, 0);

                // for (unsigned int i=0; i<local_face_dof_indices.size(); ++i)
                  internal_boundary_constraints.add_lines(local_face_dof_indices);
            }
          }
      }
      coarse_cell_of_patch++;
    }
    */


  auto sub_cell = current_patch.sub_tria.begin(0);
  for (const auto &cell : current_patch.cells)
    {
      // TODO: Find better way to get patch id
      global_to_local_cell_map[cell->active_cell_index()].push_back(std::pair<unsigned int, typename Triangulation<dim>::active_cell_iterator>(current_patch.cells[0]->active_cell_index(), sub_cell));
      // faces
      for (const auto f : cell->face_indices())
        {
          const auto face = cell->face(f);

          if (face->boundary_id() != numbers::internal_face_boundary_id)
            sub_cell->face(f)->set_boundary_id(face->boundary_id());
          // sub_cell->face(f)->set_user_index(face->boundary_id());
          else if (sub_cell->face(f)->boundary_id() !=
                   numbers::internal_face_boundary_id)
            sub_cell->face(f)->set_boundary_id(SPECIAL_NUMBER);
          // sub_cell->face(f)->set_user_index(SPECIAL_NUMBER);
        }


      // lines // useless??
      if constexpr (dim == 3)
        for (const auto l : cell->line_indices())
          {
            const auto line = cell->line(l);

            if (line->manifold_id() != numbers::flat_manifold_id)
              sub_cell->line(l)->set_manifold_id(line->manifold_id());
          }

      sub_cell++;
    }

  // refine
  // current_patch.sub_tria.refine_global(par.n_subdivisions);
}

template <int dim>
void
SLOD<dim>::assemble_global_matrix()
{
  TimerOutput::Scope t(computing_timer, "assemble global matrix");

  DoFHandler<dim> dh_fine_current_patch;
  DoFHandler<dim> dh_fine_other_patch;

  // TODO
  global_stiffness_matrix.reinit(10);

  Vector<double> phi_loc(fe_fine->n_dofs_per_cell());
  Vector<double> Aphi_loc(fe_fine->n_dofs_per_cell());

  for (auto current_patch_id : locally_owned_patches)
    {
      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];
      dh_fine_current_patch.reinit(current_patch->sub_tria);
      dh_fine_current_patch.distribute_dofs(*fe_fine);

      unsigned int i = 0;
      for (auto current_patch_cell : dh_fine_other_patch.active_cell_iterators()) {
        auto current_global_cell = current_patch->cells[i];

        current_patch_cell->get_dof_values(current_patch->basis_function_candidates[0], phi_loc);
        for (auto pair : global_to_local_cell_map[current_global_cell->active_cell_index()]) {
          auto other_patch_id = pair.first;
          auto other_patch_cell_tria = pair.second;
          const auto &other_patch = patches[other_patch_id];
          dh_fine_other_patch.reinit(other_patch.sub_tria);
          dh_fine_other_patch.distribute_dofs(*fe_fine);
          const auto other_patch_cell =
            other_patch_cell_tria->as_dof_handler_iterator(dh_fine_other_patch);


          other_patch_cell->get_dof_values(other_patch.basis_function_candidates_premultiplied[0], Aphi_loc);
          global_stiffness_matrix.add(current_patch_id, other_patch_id, phi_loc * Aphi_loc);
          
        }
        i++;
      }
    }

  //////////////////////////////
  
  // todo: do we want to do this matrix free?
  // do we want to use the same function assemble_stiffness_for_patch() for the
  // global matrix as well?

  std::unique_ptr<Quadrature<dim>> quadrature_coarse(
    std::make_unique<QGauss<dim>>(2));

  FEValues<dim> fe_values(*fe_coarse,
                          *quadrature_coarse,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int          dofs_per_cell = fe_coarse->n_dofs_per_cell();
  const unsigned int          n_q_points    = quadrature_coarse->size();
  FullMatrix<double>          cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<Tensor<1, dim>> grad_phi_u(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  // AffineConstraints<double>            constraints;
  std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim));
  Vector<double>              cell_rhs(dofs_per_cell);

  // TODO: Parameter adaptable parameters
  //  VectorTools::interpolate_boundary_values(
  //    dof_handler, 0, par.bc, constraints);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        fe_values.reinit(cell);
        // par.rhs.vector_value_list(fe_values.get_quadrature_points(),
        //                           rhs_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                grad_phi_u[k] = fe_values.shape_grad(k, q);
              }
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) +=
                      scalar_product(grad_phi_u[i], grad_phi_u[j]) *
                      fe_values.JxW(q);
                  }
                const auto comp_i =
                  fe_coarse->system_to_component_index(i).first;
                cell_rhs(i) += fe_values.shape_value(i, q) *
                               rhs_values[q][comp_i] * fe_values.JxW(q);
              }
          }


        cell->get_dof_indices(local_dof_indices);
        // TODO: Handle inhomogeneous dirichlet bc
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               global_stiffness_matrix,
                                               system_rhs);
      }
  global_stiffness_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // assemble C^T S C
}

template <int dim>
void
SLOD<dim>::assemble_stiffness_for_patch( // Patch<dim> & current_patch,
  LA::MPI::SparseMatrix &stiffness_matrix,
  const DoFHandler<dim> &dh)
{
  stiffness_matrix = 0;
  // const auto &dh = *current_patch.dh_fine;
  // TODO: stiffness_matrix should be sparse
  // TODO: avoid reallocations
  TimerOutput::Scope t(computing_timer, "Assemble patch stiffness");
  FEValues<dim>      fe_values(*fe_fine,
                          *quadrature_fine,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int          dofs_per_cell = fe_fine->n_dofs_per_cell();
  const unsigned int          n_q_points    = quadrature_fine->size();
  FullMatrix<double>          cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<Tensor<1, dim>> grad_phi_u(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  AffineConstraints<double>            local_stiffnes_constraints;

  // not needed because computed outside?
  // on the local problem we always impose homogeneous dirichlet bc
  VectorTools::interpolate_boundary_values(
    dh,
    0,
    Functions::ConstantFunction<dim, double>(0),
    local_stiffnes_constraints);
  local_stiffnes_constraints.close();

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        fe_values.reinit(cell);
        // par.rhs.vector_value_list(fe_values.get_quadrature_points(),
        //                           rhs_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                grad_phi_u[k] = fe_values.shape_grad(k, q);
              }
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) +=
                      scalar_product(grad_phi_u[i], grad_phi_u[j]) *
                      fe_values.JxW(q);
                  }
              }
          }


        cell->get_dof_indices(local_dof_indices);
        local_stiffnes_constraints.distribute_local_to_global(cell_matrix,
                                                              local_dof_indices,
                                                              stiffness_matrix);
      }
  stiffness_matrix.compress(VectorOperation::add);
}



template <int dim>
void
SLOD<dim>::solve()
{
  TimerOutput::Scope       t(computing_timer, "Solve");
  LA::MPI::PreconditionAMG prec_A;
  prec_A.initialize(global_stiffness_matrix, 1.2);

  const auto A    = linear_operator<LA::MPI::Vector>(global_stiffness_matrix);
  auto       invA = A;

  const auto amgA = linear_operator(A, prec_A);

  SolverCG<LA::MPI::Vector> cg_stiffness(par.coarse_solver_control);
  invA = inverse_operator(A, cg_stiffness, amgA);

  // Some aliases
  auto       &u = solution;
  const auto &f = system_rhs;

  u = invA * f;
  std::cout << "size of u " << u.size() << std::endl;

  constraints.distribute(u);
}

template <int dim>
void
SLOD<dim>::stabilize()
{}

template <int dim>
void
SLOD<dim>::run()
{
  print_parameters();
  make_grid();
  make_fe();
  create_patches();
  check_nested_patches();
  compute_basis_function_candidates();
  // // stabilize();
  // // now it's different and the stabilization needs the stiffness
  assemble_global_matrix();
  solve();
  // only vor vector problems
  // output_results();
  // par.convergence_table.error_from_exact(dof_handler, solution,
  // par.exact_solution);
  // // if (pcout.is_active())
  // par.convergence_table.output_table(pcout.get_stream());
}

// template<int dim>
// TransferWrapper<dim>::TransferWrapper(
//   MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<double>>
//   &transfer, unsigned int n_coarse, unsigned int n_fine) : transfer(transfer)
//   , n_coarse(n_coarse)
//   , n_fine(n_fine)
// {}

// template<int dim>
// void TransferWrapper<dim>::vmult(LinearAlgebra::distributed::Vector<double>
// &out, const LinearAlgebra::distributed::Vector<double> &in) const {
//   transfer.restrict_and_add(out, in);
// }

// template<int dim>
// void TransferWrapper<dim>::Tvmult(LinearAlgebra::distributed::Vector<double>
// &out, const LinearAlgebra::distributed::Vector<double> &in) const {
//   transfer.prolongate(out, in);
// }

// template<int dim>
// unsigned int TransferWrapper<dim>::m() const {
//   return n_coarse;
// }

// template<int dim>
// unsigned int TransferWrapper<dim>::n() const {
//   return n_fine;
// }


template class SLOD<2>;
