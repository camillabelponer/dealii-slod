#include <deal.II/base/exceptions.h>

#include <LODscalar.h>

template <int dim>
LOD<dim>::LOD(const LODParameters<dim, dim> &par)
  : par(par)
  , mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , tria(mpi_communicator)
  , dof_handler_coarse(tria)
  , dof_handler_fine(tria)
{}

template <int dim>
void
LOD<dim>::print_parameters() const
{
  TimerOutput::Scope t(computing_timer, "printing parameters");
  pcout << "Running scalar LOD diffusion problem" << std::endl;
  par.prm.print_parameters(par.output_directory + "/" + "used_parameters_" +
                             std::to_string(dim) + ".prm",
                           ParameterHandler::Short);
}

template <int dim>
void
LOD<dim>::make_fe()
{
  TimerOutput::Scope t(computing_timer, "make FE spaces");
  // fe_coarse = std::make_unique<FESystem<dim>>(FE_DGQ<dim>(0), dim);
  fe_coarse = std::make_unique<FE_DGQ<dim>>(FE_DGQ<dim>(0));
  dof_handler_coarse.distribute_dofs(*fe_coarse);

  locally_owned_dofs = dof_handler_coarse.locally_owned_dofs();
  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler_coarse);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler_coarse, constraints);
  // constraints on the boundary of the domain: UNUSED
  // TODO: check that we don't need this and delete
  VectorTools::interpolate_boundary_values(
    dof_handler_coarse,
    0,
    par.bc,
    // Functions::ConstantFunction<dim, double>(0),
    constraints);
  constraints.close();

  fe_fine =
    std::make_unique<FE_Q_iso_Q1<dim>>(FE_Q_iso_Q1<dim>(par.n_subdivisions));
  // std::make_unique<FESystem<dim>>(FE_Q_iso_Q1<dim>(par.n_subdivisions), dim);
  dof_handler_fine.distribute_dofs(*fe_fine);
  quadrature_fine = std::make_unique<Quadrature<dim>>(
    QIterated<dim>(QGauss<1>(par.p_order_on_patch), par.n_subdivisions));

  patches_pattern.reinit(dof_handler_coarse.n_dofs(),
                         dof_handler_coarse.n_dofs(),
                         locally_relevant_dofs);
  patches_pattern_fine.reinit(dof_handler_coarse.n_dofs(),
                              dof_handler_fine.n_dofs(),
                              locally_relevant_dofs);
  // DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
  // DoFTools::make_sparsity_pattern(dof_handler,
  //                                 sparsity_pattern,
  //                                 constraints,
  //                                 /*keep constrained dofs*/ false);
  // SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
  //                                            locally_owned_dofs,
  //                                            mpi_communicator,
  //                                            locally_relevant_dofs);
  // global_stiffness_matrix.reinit(locally_owned_dofs,
  //                                locally_owned_dofs,
  //                                sparsity_pattern,
  //                                mpi_communicator);
  // system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim>
void
LOD<dim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "create grid");
  GridGenerator::hyper_cube(tria);
  tria.refine_global(par.n_global_refinements);
}

template <int dim>
void
LOD<dim>::create_patches()
{
  pcout << "   creating patches";

  locally_owned_patches =
    Utilities::MPI::create_evenly_distributed_partitioning(
      mpi_communicator, tria.n_active_cells());
  // global_to_local_cell_map.resize(tria.n_active_cells());
  // patches = TrilinosWrappers::MPI::Vector(locally_owned_patches,
  //                                          mpi_communicator);
  std::vector<unsigned int> fine_dofs(fe_fine->n_dofs_per_cell());

  // Queue for patches for which neighbours should be added
  std::vector<typename DoFHandler<dim>::active_cell_iterator> patch_iterators;
  size_t size_biggest_patch = 0;
  size_t size_tiniest_patch = tria.n_active_cells();
  for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    {
      auto cell_index = cell->active_cell_index();
      // if (locally_owned_patches.is_element(cell_index))
      {
        // for each cell we create its patch and add it to the global vector
        // of patches
        auto patch = &patches.emplace_back();
        patch_iterators.clear();
        patch_iterators.push_back(cell);

        // The iterators for level l are in the range [l_start, l_end) of
        // patch_iterators
        unsigned int l_start = 0;
        unsigned int l_end   = 1;
        patch->cells.push_back(cell);
        // patch->cell_indices.set_size(tria.n_active_cells());
        patches_pattern.add(cell_index, cell_index);
        auto cell_fine = cell->as_dof_handler_iterator(dof_handler_fine);
        cell_fine->get_dof_indices(fine_dofs);
        patches_pattern_fine.add_row_entries(cell_index, fine_dofs);
        for (unsigned int l = 1; l <= par.oversampling; l++)
          {
            for (unsigned int i = l_start; i < l_end; i++)
              {
                AssertIndexRange(i, patch_iterators.size());
                for (auto ver : patch_iterators[i]->vertex_indices())
                  {
                    auto vertex = patch_iterators[i]->vertex_index(ver);
                    for (const auto &neighbour :
                         GridTools::find_cells_adjacent_to_vertex(
                           dof_handler_coarse, vertex))
                      {
                        if (!patches_pattern.exists(
                              cell_index, neighbour->active_cell_index()))
                          {
                            patch_iterators.push_back(neighbour);
                            patches_pattern.add(cell_index,
                                                neighbour->active_cell_index());
                            patches_pattern.add(cell_index,
                                                neighbour->active_cell_index());
                            auto cell_fine = neighbour->as_dof_handler_iterator(
                              dof_handler_fine);
                            cell_fine->get_dof_indices(fine_dofs);
                            patches_pattern_fine.add_row_entries(cell_index,
                                                                 fine_dofs);
                            patch->cells.push_back(neighbour);
                          }
                      }
                  }
              }
            l_start = l_end;
            l_end   = patch_iterators.size();
          }
        size_biggest_patch = std::max(size_biggest_patch, patch->cells.size());
        size_tiniest_patch = std::min(size_tiniest_patch, patch->cells.size());
      }
    }

  DynamicSparsityPattern global_sparsity_pattern;
  global_sparsity_pattern.compute_mmult_pattern(patches_pattern,
                                                patches_pattern);
  global_stiffness_matrix.reinit(locally_owned_patches,
                                 global_sparsity_pattern,
                                 mpi_communicator);
  pcout << ": done" << std::endl;

  // Utilities::MPI::min();

  // if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {
    pcout << "Number of coarse cell = " << tria.n_active_cells()
          << ", number of patches = " << patches.size()
          << " (locally owned: " << locally_owned_patches.n_elements() << ") \n"
          << "Patches size in (" << size_tiniest_patch << ", "
          << size_biggest_patch << ")" << std::endl;
  }
  // ALTERNATIVE VERSION
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
LOD<dim>::check_nested_patches()
{
  pcout << "   checking nested patches";

  for (auto current_patch_id : locally_owned_patches)
    {
      for (unsigned int i = 0; i < patches_pattern.row_length(current_patch_id);
           i++)
        {
          const auto other_patch_id =
            patches_pattern.column_number(current_patch_id, i);
          // if (other_patch_id != numbers::invalid_size_type)
          {
            if (current_patch_id == other_patch_id)
              continue;
            bool other_patch_is_contained = true;
            for (unsigned int j = 0;
                 j < patches_pattern.row_length(other_patch_id);
                 j++)
              {
                const auto cell =
                  patches_pattern.column_number(other_patch_id, j);
                // if (cell != numbers::invalid_size_type)
                {
                  if (!patches_pattern.exists(current_patch_id, cell))
                    {
                      other_patch_is_contained = false;
                      break;
                    }
                }
              }
            if (other_patch_is_contained)
              {
                AssertIndexRange(current_patch_id, patches.size());
                patches[current_patch_id].contained_patches++;
              }
          }
        }
    }

  // for (auto current_patch_id : locally_owned_patches)
  //   {
  //     AssertIndexRange(current_patch_id, patches.size());
  //     auto       current_patch    = &patches[current_patch_id];
  //     const auto current_cell_set = current_patch->cell_indices;
  //     for (auto cell_to_check : current_cell_set)
  //       {
  //         if (!(cell_to_check == current_patch_id))
  //           {
  //             AssertIndexRange(cell_to_check, patches.size());
  //             auto set_to_check = patches[cell_to_check].cell_indices;
  //             set_to_check.subtract_set(current_cell_set);
  //             if (set_to_check.is_empty())
  //               current_patch->contained_patches++;
  //           }
  //       }
  //   }
  pcout << ": done" << std::endl;
}


template <int dim>
void
LOD<dim>::output_results()
{
  TimerOutput::Scope t(computing_timer, "Output results");

  std::vector<std::string> solution_names(dim - 1, "LOD_solution");
  std::vector<std::string> exact_solution_names(dim - 1,
                                                "coarse_exact_solution");
  // std::string solution_names = "LOD_solution";
  // std::string exact_solution_names = "exact_solution";

  auto exact_vec(solution);
  VectorTools::interpolate(dof_handler_coarse, par.exact_solution, exact_vec);
  // to be added for MPI
  // auto exact_vec_locally_relevant(locally_relevant_solution.block(0));
  // exact_vec_locally_relevant = exact_vec;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      //   dim, DataComponentInterpretation::component_is_part_of_vector);
      dim - 1,
      DataComponentInterpretation::component_is_scalar);
  DataOut<dim> data_out;

  // data_out.attach_dof_handler(dof_handler_coarse);

  // data_out.add_data_vector(solution, solution_names);
  data_out.add_data_vector(dof_handler_coarse,
                           solution,
                           solution_names,
                           // DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.add_data_vector(dof_handler_coarse,
                           exact_vec,
                           exact_solution_names,
                           //  DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  // Vector<float> subdomain(tria.n_active_cells());
  // for (unsigned int i = 0; i < subdomain.size(); ++i)
  //   subdomain(i) = tria.locally_owned_subdomain();
  // data_out.add_data_vector(subdomain, "subdomain");
  data_out.build_patches();
  const std::string filename = par.output_name + ".vtu";
  data_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                 mpi_communicator);

  std::ofstream pvd_solutions(par.output_directory + "/" + par.output_name +
                              ".pvd");
}

const unsigned int SPECIAL_NUMBER = 69;


template <int dim>
void
LOD<dim>::compute_basis_function_candidates()
{
  pcout << "   computing basis functions: ";
  TimerOutput::Scope t(computing_timer, "compute basis function");

  DoFHandler<dim> dh_coarse_patch;
  DoFHandler<dim> dh_fine_patch;

  // using VectorType = LA::MPI::Vector;
  // TODO would be nice to have LA::distributed
  using VectorType = Vector<double>;

  // need reinit in loop
  LA::MPI::SparseMatrix     patch_stiffness_matrix;
  AffineConstraints<double> internal_boundary_constraints;
  AffineConstraints<double> local_stiffnes_constraints;

  for (auto current_patch_id : locally_owned_patches)
    {
      if ((current_patch_id % 50) == 0)
        pcout << "... " << current_patch_id << "/" << patches.size();

      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];

      // create_mesh_for_patch(*current_patch);
      dh_fine_patch.reinit(current_patch->sub_tria);
      dh_fine_patch.distribute_dofs(*fe_fine);
      // current_patch->dh_fine =
      // std::make_unique<DoFHandler<dim>>(dh_fine_patch);
      dh_coarse_patch.reinit(current_patch->sub_tria);
      dh_coarse_patch.distribute_dofs(*fe_coarse);

      auto   Ndofs_coarse = dh_coarse_patch.n_dofs();
      auto   Ndofs_fine   = dh_fine_patch.n_dofs();
      double h            = dh_fine_patch.begin_active()->diameter();
      // h /= (par.n_subdivisions + 1);

      IndexSet relevant_dofs;
      DoFTools::extract_locally_active_dofs(dh_fine_patch, relevant_dofs);

      {
        // constraints on the node that are on the boundary of the patch but not
        // on the boundary of the domain
        internal_boundary_constraints.clear();
        internal_boundary_constraints.reinit(relevant_dofs);
        // !!!!! is it ok to pass dh_fine_patch as copy? isn't it highly stupid?
        VectorTools::interpolate_boundary_values(
          dh_fine_patch,
          SPECIAL_NUMBER,
          Functions::ZeroFunction<dim, double>(1),
          internal_boundary_constraints);
        internal_boundary_constraints.close();
      }
      {
        // constraints on the node that are on the boundary of the patch AND
        // on the boundary of the domain, in which case the par.bc wins
        local_stiffnes_constraints.clear();
        local_stiffnes_constraints.reinit(relevant_dofs);
        VectorTools::interpolate_boundary_values(dh_fine_patch,
                                                 0,
                                                 par.bc,
                                                 local_stiffnes_constraints);
        local_stiffnes_constraints.close();
      }
      {
        // DynamicSparsityPattern dsp(relevant_dofs);
        // auto                   owned_dofs = relevant_dofs;
        auto                   owned_dofs = dh_fine_patch.locally_owned_dofs();
        DynamicSparsityPattern dsp(relevant_dofs);
        DoFTools::make_sparsity_pattern(dh_fine_patch,
                                        dsp,
                                        internal_boundary_constraints,
                                        true);
        // false);
        DoFTools::make_sparsity_pattern(dh_fine_patch,
                                        dsp,
                                        local_stiffnes_constraints,
                                        true);
        // false);
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   owned_dofs,
                                                   mpi_communicator,
                                                   relevant_dofs);
        patch_stiffness_matrix.clear();
        patch_stiffness_matrix.reinit(owned_dofs,
                                      owned_dofs,
                                      dsp,
                                      mpi_communicator);
      }

      assemble_stiffness_for_patch( //*current_patch,
        patch_stiffness_matrix,
        dh_fine_patch,
        local_stiffnes_constraints);
      // internal_boundary_constraints);

      // TODO: cahnge solver
      const auto A  = linear_operator<VectorType>(patch_stiffness_matrix);
      const auto A0 = // S
        constrained_linear_operator<VectorType>(internal_boundary_constraints,
                                                A);
      auto A0_inv = A0;

      SolverCG<VectorType> cg_A(par.fine_solver_control);
      A0_inv = inverse_operator(A0, cg_A);

      // create projection matrix from fine to coarse cell (DG)
      FullMatrix<double> projection_matrix(fe_coarse->n_dofs_per_cell(),
                                           fe_fine->n_dofs_per_cell());
      FETools::get_projection_matrix(*fe_fine, *fe_coarse, projection_matrix);

      // averaging (inverse of P0 mass matrix)
      VectorType valence_coarse(Ndofs_coarse);
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
          // should be locally owned ?
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

      const auto project = [&](auto &dst, const auto &src) {
        VectorType vec_local_coarse(fe_coarse->n_dofs_per_cell());
        VectorType vec_local_fine(fe_fine->n_dofs_per_cell());
        VectorType weights(fe_coarse->n_dofs_per_cell());

        for (const auto &cell : current_patch->sub_tria.active_cell_iterators())
          // should be locally_owned ?
          {
            const auto cell_coarse =
              cell->as_dof_handler_iterator(dh_coarse_patch);
            const auto cell_fine = cell->as_dof_handler_iterator(dh_fine_patch);

            cell_coarse->get_dof_values(src, vec_local_coarse);

            cell_coarse->get_dof_values(valence_coarse, weights);
            vec_local_coarse.scale(weights);

            projection_matrix.Tvmult(vec_local_fine, vec_local_coarse);

            cell_fine->distribute_local_to_global(vec_local_fine, dst);
          }
      };

      // Specialization of projection for the case where src is the P0 basis
      // function of a single cell Works only for P0 coarse elements
      const auto project_cell = [&](auto &dst, const auto &cell) {
        AssertDimension(fe_coarse->n_dofs_per_cell(), 1);
        // TODO: vector problem requires dim, not 1!
        VectorType vec_local_coarse(fe_coarse->n_dofs_per_cell());
        VectorType vec_local_fine(fe_fine->n_dofs_per_cell());
        VectorType weights(fe_coarse->n_dofs_per_cell());

        const auto cell_coarse = cell->as_dof_handler_iterator(dh_coarse_patch);
        const auto cell_fine   = cell->as_dof_handler_iterator(dh_fine_patch);

        // cell_coarse->get_dof_values(src, vec_local_coarse);
        vec_local_coarse[0] = 1.0;

        cell_coarse->get_dof_values(valence_coarse, weights);
        vec_local_coarse.scale(weights);

        projection_matrix.Tvmult(vec_local_fine, vec_local_coarse);

        cell_fine->distribute_local_to_global(vec_local_fine, dst);
      };

      // we now compute c_loc_i = S^-1 P^T (P S^-1 P^T)^-1 e_i
      // where e_i is the indicator function of the patch

      VectorType         P_e_i(Ndofs_fine);
      VectorType         u_i(Ndofs_fine);
      VectorType         e_i(Ndofs_coarse); // reused also as temporary vector
      VectorType         triple_product_inv_e_i(Ndofs_coarse);
      VectorType         c_i(Ndofs_fine);
      VectorType         Ac_i(Ndofs_fine);
      FullMatrix<double> triple_product(Ndofs_coarse);
      FullMatrix<double> A0_inv_P(Ndofs_fine, Ndofs_coarse);

      for (auto coarse_cell : dh_coarse_patch.active_cell_iterators())
        // for (unsigned int i = 0; i < Ndofs_coarse; ++i)
        {
          auto i = coarse_cell->active_cell_index();
          // e_i    = 0.0;
          // e_i[i] = 1.0;
          P_e_i = 0.0;
          u_i   = 0.0;

          // project(P_e_i, e_i);
          project_cell(P_e_i, coarse_cell);

          u_i = A0_inv * P_e_i;
          e_i = 0.0;
          projectT(e_i, u_i);
          for (unsigned int j = 0; j < Ndofs_coarse; j++)
            {
              triple_product(j, i) = e_i[j];
            }
          for (unsigned int j = 0; j < Ndofs_fine; j++)
            {
              A0_inv_P(j, i) = u_i[j];
            }
        }
      triple_product.gauss_jordan();

      std::vector<VectorType> candidates;
      std::vector<VectorType> Palpha_i;
      VectorType              Pa_i(Ndofs_fine);

      unsigned int N_candidates = 1;

      if (par.LOD_stabilization)
        {
          N_candidates = Ndofs_coarse;
        }
      // if we are not stabilizing then we only take the first candiadates
      // 0 is the index of the central cell
      // (this is also the central dof because we use P0 elements)

      for (unsigned int index = 0; index < N_candidates; ++index)
        {
          c_i                    = 0.0;
          e_i                    = 0.0;
          triple_product_inv_e_i = 0.0;

          e_i[index] = 1.0;
          triple_product.vmult(triple_product_inv_e_i, e_i);
          triple_product_inv_e_i /= h;

          A0_inv_P.vmult(c_i, triple_product_inv_e_i);
          c_i /= c_i.l2_norm();

          candidates.push_back(c_i);
          Pa_i = 0;
          project(Pa_i, triple_product_inv_e_i);
          Palpha_i.push_back(Pa_i);
        }

      Assert(candidates.size() > 0, ExcInternalError());

      const auto stabilize = [&](
                               Vector<double> &                   dst,
                               const std::vector<Vector<double>> &candidates) {
        unsigned int N_other_phis = candidates.size() - 1;
        if (N_other_phis > 0 &&
            (N_other_phis > current_patch->contained_patches))
          {
            // std::cout << "stabilizing..." << std::endl;
            FullMatrix<double> B(Ndofs_fine, N_other_phis);

            VectorType B_0 = A * candidates[0];
            B_0 += Palpha_i[0];

            for (unsigned int col = 0; col < Palpha_i.size() - 1; ++col)
              {
                VectorType B_i = A * candidates[col + 1];
                B_i += Palpha_i[col + 1];
                for (unsigned int row = 0; row < B_i.size(); ++row)
                  {
                    B.set(row, col, B_i[row]);
                  }
              }

            FullMatrix<double> BTB(N_other_phis, N_other_phis);
            B.Tmmult(BTB, B);
            LAPACKFullMatrix<double> SVD(N_other_phis, N_other_phis);
            SVD = BTB;
            SVD.compute_inverse_svd();
            for (unsigned int k = 0; k < current_patch->contained_patches; ++k)
              {
                SVD.remove_row_and_column(SVD.m() - 1, SVD.n() - 1);
              }
            // SVD.grow_or_shrink(N_other_phis);
            unsigned int considered_candidates = SVD.m();
            Assert((N_other_phis - current_patch->contained_patches == SVD.n()),
                   ExcInternalError());
            Assert(considered_candidates == SVD.n(), ExcInternalError());

            VectorType d_i(considered_candidates);
            d_i = 0;
            // FullMatrix<double> Matrix_rhs(SVD.m(),Ndofs_coarse - 1);
            // LAPACKFullMatrix<double> Blapack(Ndofs_fine, Ndofs_coarse - 1);
            // Blapack = B;
            // VectorType
            // SVD.mTmult(Matrix_rhs, Blapack);
            // Matrix_rhs.vmult(d_i, B_0);
            VectorType BTB_0(N_other_phis);
            B.Tvmult(BTB_0, B_0);
            BTB_0.grow_or_shrink(considered_candidates);
            SVD.vmult(d_i, BTB_0);

            dst = candidates[0];
            for (unsigned int index = 0; index < considered_candidates; ++index)
              {
                dst += d_i[index] * candidates[index];
              }
            dst /= dst.l2_norm();
          }
        else
          dst = candidates[0];
      };

      Vector<double> selected_basis_function;
      stabilize(selected_basis_function, candidates);

      current_patch->basis_function               = selected_basis_function;
      Ac_i                                        = 0;
      Ac_i                                        = A * selected_basis_function;
      current_patch->basis_function_premultiplied = Ac_i;

      dh_fine_patch.clear();
    }

  pcout << " done" << std::endl;
}

template <int dim>
void
LOD<dim>::create_mesh_for_patch(Patch<dim> &current_patch)
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

  auto sub_cell = current_patch.sub_tria.begin(0);
  for (const auto &cell : current_patch.cells)
    {
      // TODO: Find better way to get patch id
      // global_to_local_cell_map[cell->active_cell_index()].push_back(
      //   std::pair<unsigned int,
      //             typename Triangulation<dim>::active_cell_iterator>(
      //     current_patch.cells[0]->active_cell_index(), sub_cell));
      // faces
      for (const auto f : cell->face_indices())
        {
          const auto face = cell->face(f);
          // if we are at boundary of patch AND domain -> keep boundary_id
          if (face->at_boundary())
            sub_cell->face(f)->set_boundary_id(face->boundary_id());
          // if the face is not at the boundary of the domain, is it at the
          // boundary of the patch?
          else if (sub_cell->face(f)->boundary_id() !=
                   numbers::internal_face_boundary_id)
            // it's not at te boundary of the patch -> then is our "internal
            // boundary"
            sub_cell->face(f)->set_boundary_id(SPECIAL_NUMBER);
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
LOD<dim>::assemble_global_matrix()
{
  TimerOutput::Scope t(computing_timer, "assemble global matrix");

  DoFHandler<dim> dh_fine_current_patch;

  // auto     lod = dh_fine_current_patch.locally_owned_dofs();
  // TODO: for mpi should not allocate all cols and rows->create partitioning
  // like we do for global_stiffness_matrix.reinit(..)
  basis_matrix.reinit(patches_pattern_fine.nonempty_rows(),
                      patches_pattern_fine.nonempty_cols(),
                      patches_pattern_fine,
                      mpi_communicator);

  // if we don't want to use the operator to compute the global_stiffnes matrix
  // as a multiplication then we need the transpose of the patches_pattern_fine
  // and in this case the matrix premultiplied_basis_matrix will saved already
  // in the transposed form
  DynamicSparsityPattern identity(patches_pattern_fine.nonempty_rows());
  for (unsigned int i = 0; i < patches_pattern_fine.n_rows(); ++i)
    identity.add(i, i);
  DynamicSparsityPattern patches_pattern_fine_T;
  patches_pattern_fine_T.compute_Tmmult_pattern(patches_pattern_fine, identity);
  premultiplied_basis_matrix.reinit(patches_pattern_fine_T.nonempty_rows(),
                                    patches_pattern_fine_T.nonempty_cols(),
                                    patches_pattern_fine_T,
                                    mpi_communicator);

  /* premultiplied_basis_matrix.reinit(
      patches_pattern_fine.nonempty_rows(),
      patches_pattern_fine.nonempty_cols(),
      patches_pattern_fine,
      mpi_communicator);
      */
  basis_matrix               = 0.0;
  premultiplied_basis_matrix = 0.0;
  system_rhs.reinit(patches_pattern_fine.nonempty_rows(), mpi_communicator);
  LA::MPI::Vector rhs_values(patches_pattern_fine.nonempty_cols(),
                             mpi_communicator);
  rhs_values = 0.0;

  Vector<double>            phi_loc(fe_fine->n_dofs_per_cell());
  std::vector<unsigned int> global_dofs(fe_fine->n_dofs_per_cell());

  for (auto current_patch_id : locally_owned_patches)
    {
      const auto current_patch = &patches[current_patch_id];
      dh_fine_current_patch.reinit(current_patch->sub_tria);
      dh_fine_current_patch.distribute_dofs(*fe_fine);

      for (auto iterator_to_cell_in_current_patch :
           dh_fine_current_patch.active_cell_iterators())
        {
          auto iterator_to_cell_global =
            current_patch
              ->cells[iterator_to_cell_in_current_patch->active_cell_index()]
              ->as_dof_handler_iterator(dof_handler_fine);
          iterator_to_cell_global->get_dof_indices(global_dofs);

          iterator_to_cell_in_current_patch->get_dof_values(
            current_patch->basis_function, phi_loc);
          // AssertDimension(global_dofs.size(), phi_loc.size())
          basis_matrix.set(current_patch_id,
                           phi_loc.size(),
                           global_dofs.data(),
                           phi_loc.data());

          iterator_to_cell_in_current_patch->get_dof_values(
            current_patch->basis_function_premultiplied, phi_loc);
          AssertDimension(global_dofs.size(), phi_loc.size())
            // premultiplied_basis_matrix.set(current_patch_id,
            //                                phi_loc.size(),
            //                                global_dofs.data(),
            //                                phi_loc.data());
            // if the matrix is already transposed we need to loop to add the
            // elements
            for (unsigned int idx = 0; idx < phi_loc.size(); ++idx)
          {
            premultiplied_basis_matrix.set(global_dofs.data()[idx],
                                           current_patch_id,
                                           phi_loc.data()[idx]);
          }
        }
    }
  basis_matrix.compress(VectorOperation::insert);
  premultiplied_basis_matrix.compress(VectorOperation::insert);
  // std::cout << "     basis matrix frobenius norm = "
  //           << basis_matrix.frobenius_norm() << std::endl;
  // std::cout << "     premultiplied basis matrix frobenius norm = "
  //           << premultiplied_basis_matrix.frobenius_norm() << std::endl;

  // TODO: the following line is done twice: save M or compute here also
  // lod_solution needed to compare
  // const auto M = linear_operator<LA::MPI::Vector>(basis_matrix);
  // VectorTools::interpolate(dof_handler_fine, par.rhs, rhs_values);
  // basis_matrix.vmult(system_rhs, rhs_values);
  // system_rhs = M * rhs_values;
  // VectorTools::interpolate(dof_handler_coarse, par.rhs, system_rhs);
  // std::cout << "     rhs l2 norm = " << system_rhs.l2_norm() << std::endl;

  basis_matrix.mmult(global_stiffness_matrix, premultiplied_basis_matrix);

  // std::cout << "     global stiffness matrix frobenius norm = "
  //           << global_stiffness_matrix.frobenius_norm() << std::endl;

  global_stiffness_matrix.compress(VectorOperation::add);
  // system_rhs.compress(VectorOperation::add);

  ////////
  // DOES NOT WORK
  // This adds values that are on the boundary of patches multiple times. I
  // think there is no way to avoid this but to use a global fine dof handler.

  /*
  DoFHandler<dim> dh_fine_current_patch;
  DoFHandler<dim> dh_fine_other_patch;

  Vector<double>                       phi_loc(fe_fine->n_dofs_per_cell());
  Vector<double>                       Aphi_loc(fe_fine->n_dofs_per_cell());
  std::vector<types::global_dof_index> local_dof_indices;

  for (auto current_patch_id : locally_owned_patches)
    {
      AssertIndexRange(current_patch_id, patches.size());
      const auto current_patch = &patches[current_patch_id];
      dh_fine_current_patch.reinit(current_patch->sub_tria);
      dh_fine_current_patch.distribute_dofs(*fe_fine);

      unsigned int current_cell_id = 0; // != current_patch_id because i loop
  over the cells in the same patch for (auto iterator_to_cell_in_current_patch :
           dh_fine_current_patch.active_cell_iterators())
        {
          FullMatrix<double>
  local_stiffness_matrix(patches_pattern.row_length(current_cell_id),
  patches_pattern.row_length(current_cell_id)); auto iterator_to_cell_global =
  current_patch->cells[current_cell_id];

          iterator_to_cell_in_current_patch->get_dof_values(
            current_patch->basis_function, phi_loc);
          auto j = 0;

          for (auto pair : global_to_local_cell_map.at(
                 iterator_to_cell_global->active_cell_index()))
            {
              auto        other_patch_id        = pair.first;
              AssertIndexRange(other_patch_id, patches.size());
              auto        other_patch_cell_tria = pair.second;
              const auto &other_patch           = patches[other_patch_id];
              dh_fine_other_patch.reinit(other_patch.sub_tria);
              dh_fine_other_patch.distribute_dofs(*fe_fine);
              const auto iterator_to_cell_in_other_patch =
                other_patch_cell_tria->as_dof_handler_iterator(
                  dh_fine_other_patch);


              iterator_to_cell_in_other_patch->get_dof_values(
                other_patch.basis_function_premultiplied,
                Aphi_loc);
              //local_stiffness_matrix(current_patch_id, other_patch_id) +=
               local_stiffness_matrix(0, j) += phi_loc * Aphi_loc;
              j++;
            }

          local_dof_indices.clear();
          for (unsigned int j = 0;
               j < patches_pattern.row_length(current_cell_id);
               j++)
               {
            local_dof_indices.push_back(
              patches_pattern.column_index(current_cell_id, j));
               }
         constraints.distribute_local_to_global(local_stiffness_matrix,
                                                 local_dof_indices,
                                                 global_stiffness_matrix);

          current_cell_id++;
        }
    }
  */
}

template <int dim>
void
LOD<dim>::assemble_stiffness_for_patch( // Patch<dim> & current_patch,
  LA::MPI::SparseMatrix &    stiffness_matrix,
  const DoFHandler<dim> &    dh,
  AffineConstraints<double> &local_stiffnes_constraints)
{
  // const auto &dh = *current_patch.dh_fine;
  // TimerOutput::Scope t(
  //   computing_timer,
  //   "compute basis functions: Assemble patch stiffness");
  stiffness_matrix = 0;
  FEValues<dim> fe_values(*fe_fine,
                          *quadrature_fine,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int          dofs_per_cell = fe_fine->n_dofs_per_cell();
  const unsigned int          n_q_points    = quadrature_fine->size();
  FullMatrix<double>          cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<Tensor<1, dim>> grad_phi_u(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  // AffineConstraints<double>            local_stiffnes_constraints;

  // IndexSet relevant_dofs;
  // DoFTools::extract_locally_relevant_dofs(dh, relevant_dofs);
  // local_stiffnes_constraints.reinit(relevant_dofs);


  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        fe_values.reinit(cell);
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
LOD<dim>::solve()
{
  TimerOutput::Scope       t(computing_timer, "Solve");
  LA::MPI::PreconditionAMG prec_A;
  prec_A.initialize(global_stiffness_matrix, 1.2);

  const auto C = linear_operator<LA::MPI::Vector>(basis_matrix);
  // const auto AM =
  // linear_operator<LA::MPI::Vector>(premultiplied_basis_matrix);
  // const auto AMT = transpose_operator(AM);
  // const auto A = M*AMT;
  const auto A    = linear_operator<LA::MPI::Vector>(global_stiffness_matrix);
  auto       invA = A;

  // TODO: oversampling = 0 global stiffness is not well defined

  const auto amgA = linear_operator(A, prec_A);

  SolverCG<LA::MPI::Vector> cg_stiffness(par.coarse_solver_control);
  invA = inverse_operator(A, cg_stiffness, amgA);

  system_rhs = C * fem_rhs;
  pcout << "     rhs l2 norm = " << system_rhs.l2_norm() << std::endl;


  // Some aliases
  auto &      u = solution;
  const auto &f = system_rhs;

  u = invA * f;
  pcout << "   size of u " << u.size() << std::endl;
  constraints.distribute(u);
}

template <int dim>
void
LOD<dim>::solve_fem_problem() //_and_compare() // const
{
  TimerOutput::Scope t(computing_timer, "solve fine FEM");

  const auto &dh = dof_handler_fine;

  auto     locally_owned_dofs = dh.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dh, locally_relevant_dofs);
  FEValues<dim> fe_values(*fe_fine,
                          *quadrature_fine,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // create sparsity pattern fr global fine matrix
  AffineConstraints<double> fem_constraints(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dh, fem_constraints);
  VectorTools::interpolate_boundary_values(
    dh,
    0,
    // Functions::ConstantFunction<dim, double>(0),
    par.bc,
    fem_constraints);
  fem_constraints.close();
  DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dh, sparsity_pattern, fem_constraints, false);
  SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);

  LA::MPI::SparseMatrix fem_stiffness_matrix;
  fem_stiffness_matrix.reinit(locally_owned_dofs,
                              locally_owned_dofs,
                              sparsity_pattern,
                              mpi_communicator);
  // locally_relevant_solution.reinit(locally_owned_dofs,
  //                                  locally_relevant_dofs,
  //                                  mpi_communicator);
  fem_rhs.reinit(locally_owned_dofs, mpi_communicator);
  fem_solution.reinit(locally_owned_dofs, mpi_communicator);

  LA::MPI::Vector locally_relevant_solution(locally_owned_dofs,
                                            locally_relevant_dofs,
                                            mpi_communicator);
  // LA::MPI::Vector fem_rhs(locally_owned_dofs, mpi_communicator);
  // LA::MPI::Vector fem_solution(locally_owned_dofs, mpi_communicator);

  // assemble fine global matrix
  const unsigned int          dofs_per_cell = fe_fine->n_dofs_per_cell();
  const unsigned int          n_q_points    = quadrature_fine->size();
  FullMatrix<double>          cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<Tensor<1, dim>> grad_phi_u(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>                  rhs_values(n_q_points);
  Vector<double>                       cell_rhs(dofs_per_cell);

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);
        // par.rhs.vector_value_list(fe_values.get_quadrature_points(),
        par.rhs.value_list(fe_values.get_quadrature_points(), rhs_values);
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
                // const auto comp_i =
                //   fe_coarse->system_to_component_index(i).first;
                // cell_rhs(i) += fe_values.shape_value(i, q) *
                //                rhs_values[q][comp_i] * fe_values.JxW(q);
                cell_rhs(i) += fe_values.shape_value(i, q) * rhs_values[q] *
                               fe_values.JxW(q);
              }
          }


        cell->get_dof_indices(local_dof_indices);
        fem_constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   fem_stiffness_matrix,
                                                   fem_rhs);
      }

  fem_stiffness_matrix.compress(VectorOperation::add);
  fem_rhs.compress(VectorOperation::add);

  // std::cout << "     fine stiffness matrix frobenius norm = "
  //           << fem_stiffness_matrix.frobenius_norm() << std::endl;
  pcout << "     fem rhs l2 norm = " << fem_rhs.l2_norm() << std::endl;

  // solve
  LA::MPI::PreconditionAMG prec_Sh;
  prec_Sh.initialize(fem_stiffness_matrix, 1.2);

  const auto Sh    = linear_operator<LA::MPI::Vector>(fem_stiffness_matrix);
  auto       invSh = Sh;

  const auto amg = linear_operator(Sh, prec_Sh);

  SolverCG<LA::MPI::Vector> cg_stiffness(par.fine_solver_control);
  invSh = inverse_operator(Sh, cg_stiffness, amg);

  fem_solution = invSh * fem_rhs;
  pcout << "   size of fem u " << fem_solution.size() << std::endl;
  fem_constraints.distribute(fem_solution);
}

template <int dim>
void
LOD<dim>::compare_fem_lod()
{
  computing_timer.enter_subsection("compare FEM vs LOD");
  const auto &dh = dof_handler_fine;
  // compare
  // 1
  // parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> transfer(
  // dof_handler);
  // transfer.interpolate(fem_solution);
  // fem_constraints.distribute(fem_solution);
  // locally_relevant_solution = fem_solution;
  // 2
  // InterGridMap<DoFHandler<dim> > coarse_to_fine_map;
  // coarse_to_fine_map.make_mapping(dof_handler, dh_fine);
  // 3
  // VectorTools::interpolate_to_different_mesh(coarse_to_fine_map,
  //                                           solution,
  //                                           constraints,
  //                                           dst);
  // FETools::interpolate(dof_handler_coarse, solution, dh,
  // lod_solution);
  // 4

  LA::MPI::Vector lod_solution(patches_pattern_fine.nonempty_cols(),
                               mpi_communicator);
  lod_solution = 0;

  const auto C  = linear_operator<LA::MPI::Vector>(basis_matrix);
  const auto CT = transpose_operator(C);
  lod_solution  = CT * solution;

  par.convergence_table_compare.difference(dh, fem_solution, lod_solution);
  par.convergence_table_FEM.error_from_exact(dh,
                                             fem_solution,
                                             par.exact_solution);
  par.convergence_table_LOD.error_from_exact(dh,
                                             lod_solution,
                                             par.exact_solution);

  computing_timer.leave_subsection();
  computing_timer.enter_subsection("fine output");

  // output fem solution
  std::vector<std::string> fem_names(1, "fem_solution");
  std::vector<std::string> exact_solution_names(1, "exact_solution_fine");
  std::vector<std::string> lod_names(1, "lod_solution_fine");

  auto exact_vec(fem_solution);
  VectorTools::interpolate(dh, par.exact_solution, exact_vec);
  // to be added for MPI
  // auto exact_vec_locally_relevant(locally_relevant_solution.block(0));
  // exact_vec_locally_relevant = exact_vec;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      //   dim, DataComponentInterpretation::component_is_part_of_vector);
      dim - 1,
      DataComponentInterpretation::component_is_scalar);
  DataOut<dim> data_out;

  data_out.add_data_vector(dh,
                           fem_solution,
                           fem_names,
                           // DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.add_data_vector(dh,
                           exact_vec,
                           exact_solution_names,
                           //  DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.add_data_vector(dh,
                           lod_solution,
                           lod_names,
                           //  DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.build_patches();
  const std::string filename = par.output_name + "_fine.vtu";
  data_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                 mpi_communicator);

  std::ofstream pvd_solutions(par.output_directory + "/" + par.output_name +
                              "_fine.pvd");


  computing_timer.leave_subsection();
}

template <int dim>
void
LOD<dim>::initialize_patches()
{
  TimerOutput::Scope t(computing_timer, "Initialize patches");
  create_patches();
  // MPI Barrier
  check_nested_patches();
  pcout << "   creating mesh for patches";
  for (auto current_patch_id : locally_owned_patches)
    {
      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];
      create_mesh_for_patch(*current_patch);
    }
  pcout << ": done" << std::endl;

  // if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ||
  //     Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
  //   {
  //     const std::string filename(par.output_directory + "/patches.txt");
  //     std::ofstream     file;
  //     file.open(filename);
  //     file << "printing the sparsity pattern: [global_cell_id] = {cells}"
  //          << std::endl;
  //     // for (unsigned int cell = 0; cell < tria.n_active_cells(); ++cell)
  //     for (const auto &cell_it : tria.active_cell_iterators())
  //       {
  //         auto cell = cell_it->active_cell_index();
  //         file << "- cell " << cell << " (baricenter " <<
  //         cell_it->barycenter()
  //              << ") is connected to patches/cells: {";
  //         for (unsigned int j = 0; j < patches_pattern.row_length(cell); j++)
  //           {
  //             file << patches_pattern.column_number(cell, j) << " ";
  //           }
  //         file << "}. contained patches = " <<
  //         patches[cell].contained_patches
  //              << std::endl;
  //       }
  //     file.close();
  //   }
}

template <int dim>
void
LOD<dim>::run()
{
  print_parameters();
  make_grid();
  make_fe();
  initialize_patches();

  compute_basis_function_candidates();
  assemble_global_matrix();
  solve_fem_problem();
  solve();
  compare_fem_lod();

  output_results();
  // par.convergence_table_LOD.error_from_exact(dof_handler_coarse,
  //                                             solution,
  //                                             par.exact_solution);
  if (pcout.is_active())
    {
      pcout << "LOD vs exact solution (fine mesh)" << std::endl;
      par.convergence_table_LOD.output_table(pcout.get_stream());
      pcout << "FEM vs exact solution (fine mesh)" << std::endl;
      par.convergence_table_FEM.output_table(pcout.get_stream());
      pcout << "LOD vs FEM (fine mesh)" << std::endl;
      par.convergence_table_compare.output_table(pcout.get_stream());
    }
}


template class LOD<2>;
