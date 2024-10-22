#include <deal.II/base/exceptions.h>

#include <LOD.h>
#include <LODtools.h>

const unsigned int SPECIAL_NUMBER = 99;
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


template <int dim, int spacedim>
LOD<dim, spacedim>::LOD(const LODParameters<dim, spacedim> &par)
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

template <int dim, int spacedim>
void
LOD<dim, spacedim>::print_parameters() const
{
  TimerOutput::Scope t(computing_timer, "0: Printing parameters");
  if constexpr (spacedim == 1)
    pcout << "Running LOD Diffusion problem in " << dim << "D" << std::endl;
  else // spacedim == dim (3D case not yet implemented)
    pcout << "Running LOD Elasticity problem in " << dim << "D" << std::endl;

  par.prm.print_parameters(par.output_directory + "/" + "used_parameters_" +
                             std::to_string(dim) + ".prm",
                           ParameterHandler::Short);
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::make_fe()
{
  TimerOutput::Scope t(computing_timer, "0: Make FE spaces");
  fe_coarse = std::make_unique<FESystem<dim>>(FE_DGQ<dim>(0), spacedim);
  // fe_coarse = std::make_unique<FE_DGQ<dim>>(FE_DGQ<dim>(0));
  dof_handler_coarse.distribute_dofs(*fe_coarse);

  locally_owned_dofs = dof_handler_coarse.locally_owned_dofs();
  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler_coarse);

  // constraints on the boundary of the domain
  coarse_boundary_constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler_coarse,
                                          coarse_boundary_constraints);
  VectorTools::interpolate_boundary_values(dof_handler_coarse,
                                           0,
                                           par.bc,
                                           coarse_boundary_constraints);
  coarse_boundary_constraints.close();

  fe_fine =
    // std::make_unique<FE_Q_iso_Q1<dim>>(FE_Q_iso_Q1<dim>(par.n_subdivisions));
    std::make_unique<FESystem<dim>>(FE_Q_iso_Q1<dim>(par.n_subdivisions),
                                    spacedim);
  dof_handler_fine.distribute_dofs(*fe_fine);
  quadrature_fine = std::make_unique<Quadrature<dim>>(
    QIterated<dim>(QGauss<1>(2), par.n_subdivisions));

  patches_pattern.reinit(dof_handler_coarse.n_dofs(),
                         dof_handler_coarse.n_dofs(),
                         locally_relevant_dofs);
  patches_pattern_fine.reinit(dof_handler_coarse.n_dofs(),
                              dof_handler_fine.n_dofs(),
                              locally_relevant_dofs);

  bool_dof_mask = create_bool_dof_mask_Q_iso_Q1(*fe_fine,
                                                *quadrature_fine,
                                                par.n_subdivisions);
  // MPI: instead of having every processor compute it we could just comunicate
  // it
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "0: Make grid");
  GridGenerator::hyper_cube(tria);
  tria.refine_global(par.n_global_refinements);

  locally_owned_patches =
    Utilities::MPI::create_evenly_distributed_partitioning(
      mpi_communicator, tria.n_global_active_cells());
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::create_patches()
{
  TimerOutput::Scope        t(computing_timer, "1: Create Patches");
  std::vector<unsigned int> fine_dofs(fe_fine->n_dofs_per_cell());

  // Queue for patches for which neighbours should be added
  std::vector<typename DoFHandler<dim>::active_cell_iterator> patch_iterators;
  size_t size_biggest_patch = 0;
  size_t size_tiniest_patch = tria.n_active_cells();

  double       H                = pow(0.5, par.n_global_refinements);
  unsigned int N_cells_per_line = (int)1 / H;
  std::vector<typename DoFHandler<dim>::active_cell_iterator> ordered_cells;
  ordered_cells.resize(tria.n_active_cells());
  std::vector<std::vector<unsigned int>> cells_in_patch;
  cells_in_patch.resize(tria.n_active_cells());

  for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    {
      const double x = cell->barycenter()(0);
      const double y = cell->barycenter()(1);

      // const unsigned int x_i = (int)floor(x/H);
      // const unsigned int y_i = (int)floor(y/H);
      const unsigned int vector_cell_index =
        (int)floor(x / H) + N_cells_per_line * (int)floor(y / H);
      ordered_cells[vector_cell_index] = cell;

      std::vector<unsigned int> connected_indeces;
      connected_indeces.push_back(vector_cell_index);
    
      for (int l_row = -par.oversampling;
           l_row <= static_cast<int>(par.oversampling);
           ++l_row)
        {
          double x_j = x + l_row * H;
          if (x_j > 0 && x_j < 1) // domain borders
            {
              for (int l_col = -par.oversampling;
                   l_col <= static_cast<int>(par.oversampling);
                   ++l_col)
                {
                  const double y_j = y + l_col * H;
                  if (y_j > 0 && y_j < 1)
                    {
                      const unsigned int vector_cell_index_j =
                        (int)floor(x_j / H) +
                        N_cells_per_line * (int)floor(y_j / H);
                      if (vector_cell_index != vector_cell_index_j)
                        connected_indeces.push_back(vector_cell_index_j);
                    }
                }
            }
        }

      cells_in_patch[vector_cell_index] = connected_indeces;
    }
    
  // now looping and creating the patches
  for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    {
      const auto vector_cell_index =
        (int)floor(cell->barycenter()(0) / H) +
        N_cells_per_line * (int)floor(cell->barycenter()(1) / H);
      auto cell_index = cell->active_cell_index();
      {
        auto patch = &patches.emplace_back();


        for (auto neighbour_ordered_index : cells_in_patch[vector_cell_index])
          {
            auto &cell_to_add = ordered_cells[neighbour_ordered_index];
            patch->cells.push_back(cell_to_add);
            patches_pattern.add(cell_index, cell_to_add->active_cell_index());
            auto cell_fine =
              cell_to_add->as_dof_handler_iterator(dof_handler_fine);
            cell_fine->get_dof_indices(fine_dofs);
            patches_pattern_fine.add_row_entries(cell_index, fine_dofs);
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

  solution.reinit(locally_owned_patches, mpi_communicator);


  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ||
      Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
    {
      pcout << "Number of coarse cell = " << tria.n_active_cells()
            << ", number of patches = " << patches.size()
            << " (locally owned: " << locally_owned_patches.n_elements()
            << ") \n"
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

template <int dim, int spacedim>
void
LOD<dim, spacedim>::check_nested_patches()
{
  TimerOutput::Scope t(computing_timer, "1: Check nested Patches");
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
}


template <int dim, int spacedim>
void
LOD<dim, spacedim>::output_results()
{
  TimerOutput::Scope t(computing_timer, "6: Output results");

  std::vector<std::string> solution_names(spacedim, "LOD_solution");
  std::vector<std::string> exact_solution_names(spacedim,
                                                "coarse_exact_solution");

  auto exact_vec(solution);
  VectorTools::interpolate(dof_handler_coarse, par.exact_solution, exact_vec);
  // to be added for MPI
  // auto exact_vec_locally_relevant(locally_relevant_solution.block(0));
  // exact_vec_locally_relevant = exact_vec;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      spacedim,
      //   DataComponentInterpretation::component_is_part_of_vector);
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


template <int dim, int spacedim>
void
LOD<dim, spacedim>::compute_basis_function_candidates()
{
  // TimerOutput::Scope t(computing_timer, "compute basis function");
  computing_timer.enter_subsection("2: compute basis function 0");
  DoFHandler<dim> dh_coarse_patch;
  DoFHandler<dim> dh_fine_patch;

  using VectorType = Vector<double>;

  // need reinit in loop
  // LA::MPI::SparseMatrix
  // SparseMatrix<double>
  // TrilinosWrappers::SparseMatrix patch_stiffness_matrix;
  TrilinosWrappers::SparseMatrix unconstrained_patch_stiffness_matrix;
  AffineConstraints<double>      internal_boundary_constraints; // not used
  AffineConstraints<double>      empty_boundary_constraints;
  empty_boundary_constraints.close();
  // AffineConstraints<double> local_stiffnes_constraints;

  // TODO: use internal and local constraints to take care fo the boundary of
  // the patch that's not on the boundary of the domain now special number is
  // set to zero so they are treated as one together

  // Table<2, bool> bool_dof_mask =
  //   create_bool_dof_mask_Q_iso_Q1(*fe_fine,
  //                                 *quadrature_fine,
  //                                 par.n_subdivisions);

  // we are assuming mesh to be created as hyper_cube l 83
  double H = pow(0.5, par.n_global_refinements);
  double h = H / (par.n_subdivisions);

  // create projection matrix from fine to coarse cell (DG)
  FullMatrix<double> projection_matrix(fe_coarse->n_dofs_per_cell(),
                                       fe_fine->n_dofs_per_cell());
  // FETools::get_projection_matrix(*fe_fine, *fe_coarse, projection_matrix);
  projection_P0_P1<dim>(projection_matrix);
  projection_matrix *= (h * h / 4);
  // this could be done via tensor product
  computing_timer.leave_subsection();
  for (auto current_patch_id : locally_owned_patches)
    {
      computing_timer.enter_subsection(
        "2: compute basis function 1: patch setup");

      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];

      // create_mesh_for_patch(*current_patch);
      dh_fine_patch.reinit(current_patch->sub_tria);
      dh_fine_patch.distribute_dofs(*fe_fine);
      // DoFRenumbering::cell_wise(dh_fine_patch, cell_order);

      dh_coarse_patch.reinit(current_patch->sub_tria);
      dh_coarse_patch.distribute_dofs(*fe_coarse);
      // DoFRenumbering::cell_wise(dh_fine_patch, cell_order);

      auto Ndofs_coarse = dh_coarse_patch.n_dofs();
      auto Ndofs_fine   = dh_fine_patch.n_dofs();

      //create_dof_indices()

// std::set<unsigned int> bd_of_patch_index{SPECIAL_NUMBER};
// std::set<unsigned int> bd_of_domain_index{0};
//       IndexSet boundary_dofs_of_patch_set =
//         DoFTools::extract_boundary_dofs(dh_fine_patch, ComponentMask(), bd_of_patch_index);

// boundary_dofs_of_patch_set.print(std::cout);
//       std::vector<unsigned int> boundary_dofs_fine;
//       boundary_dofs_of_patch_set.fill_index_vector(boundary_dofs_fine);
//       std::cout << "boundary: " ;
//       for (auto b : boundary_dofs_fine)
//       std::cout << b << " ";
//       std::cout << std::endl;

//       std::vector<unsigned int> internal_dofs_fine;
//       std::vector<unsigned int> all_dofs_fine;
//       // TODO : change, this is ugly
//       for (unsigned int i = 0; i < Ndofs_fine; ++i)
//         {
//           all_dofs_fine.push_back(i);
//           if (!boundary_dofs_of_patch_set.is_element(i))
//             internal_dofs_fine.push_back(i);
//         }
//             std::cout << "internal: " ;
//       for (auto b : internal_dofs_fine)
//       std::cout << b << " ";
//       std::cout << std::endl;

std::vector<unsigned int> internal_dofs_fine;
          std::vector<unsigned int> all_dofs_fine;
          std::vector<unsigned int> /*patch_*/boundary_dofs_fine;

          fill_dofs_indices_vector(dh_fine_patch, all_dofs_fine, internal_dofs_fine, /*patch_*/boundary_dofs_fine);
          


      std::vector<unsigned int> all_dofs_coarse(all_dofs_fine.begin(),
                                                all_dofs_fine.begin() +
                                                  Ndofs_coarse);

      unsigned int       considered_candidates = Ndofs_coarse - 1;
      const unsigned int N_boundary_dofs       = boundary_dofs_fine.size();
      const unsigned int N_internal_dofs       = internal_dofs_fine.size();

      computing_timer.leave_subsection();
      computing_timer.enter_subsection(
        "2: compute basis function 2: constraints");
      internal_boundary_constraints.clear();
      DoFTools::make_zero_boundary_constraints(dh_fine_patch,
                                               0,
                                               internal_boundary_constraints);
      internal_boundary_constraints.close();
      computing_timer.leave_subsection();
      computing_timer.enter_subsection(
        "2: compute basis function 3: sparsity pattern");
      DynamicSparsityPattern patch_sparsity_pattern(Ndofs_fine);
      DynamicSparsityPattern internal_sparsity_pattern(Ndofs_fine);

      if (false)
        {
          // option 2
          // IndexSet relevant_dofs;
          // DoFTools::extract_locally_active_dofs(dh_fine_patch,
          // relevant_dofs); 
          // DynamicSparsityPattern
          // sparsity_pattern(relevant_dofs);

          DoFTools::make_sparsity_pattern(dh_fine_patch,
                                          patch_sparsity_pattern,
                                          empty_boundary_constraints,
                                          false);
          unconstrained_patch_stiffness_matrix.clear();
          unconstrained_patch_stiffness_matrix.reinit(patch_sparsity_pattern);
          // internal_sparsity_pattern.copy_from(sparsity_patter);
        }
      else
        {
          // option 1: does the same as
          // DoFTools::make_sparsity_pattern() but also
          // considers bool_dof_mask for FE_Q_iso_Q1
          // it can be slower for small par.n_subdivisions or small
          // par.oversampling

          // TrilinosWrappers::
          // SparsityPattern patch_sparsity_pattern(Ndofs_fine, Ndofs_fine);

          std::vector<types::global_dof_index> dofs_on_this_cell(
            fe_fine->n_dofs_per_cell());

          for (const auto &cell : dh_fine_patch.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                // const unsigned int dofs_per_cell =
                //   cell->get_fe().n_dofs_per_cell();
                // dofs_on_this_cell.resize(dofs_per_cell);
                cell->get_dof_indices(dofs_on_this_cell);

                // internal_boundary_constraints.add_entries_local_to_global(
                empty_boundary_constraints.add_entries_local_to_global(
                  dofs_on_this_cell,
                  patch_sparsity_pattern,
                  true,
                  bool_dof_mask); // keep constrained entries must be true

                internal_boundary_constraints.add_entries_local_to_global(
                  dofs_on_this_cell,
                  internal_sparsity_pattern,
                  false,
                  bool_dof_mask); // here should be false
              }

          patch_sparsity_pattern.compress();
          unconstrained_patch_stiffness_matrix.clear();
          unconstrained_patch_stiffness_matrix.reinit(patch_sparsity_pattern);
        }


      computing_timer.leave_subsection();
      computing_timer.enter_subsection(
        "2: compute basis function 4: stiffness");


      if (false)
        {
          MappingQ1<dim> mapping;

          MatrixCreator::create_laplace_matrix<dim, dim>(
            mapping,
            dh_fine_patch,
            *quadrature_fine,
            unconstrained_patch_stiffness_matrix,
            nullptr,
            // internal_boundary_constraints);
            empty_boundary_constraints);
        }
      else
        {
          LA::MPI::Vector dummy;
          assemble_stiffness(unconstrained_patch_stiffness_matrix,
                             dummy,
                             dh_fine_patch,
                             // internal_boundary_constraints);
                             empty_boundary_constraints);
        }


      computing_timer.leave_subsection();
      computing_timer.enter_subsection("2: compute basis function 4b: misc");

      // averaging (inverse of P0 mass matrix)
      VectorType valence_coarse(Ndofs_coarse);
      VectorType local_identity_coarse(fe_coarse->n_dofs_per_cell());
      local_identity_coarse = 1.0;

      for (const auto &cell : dh_coarse_patch.active_cell_iterators())
        cell->distribute_local_to_global(local_identity_coarse, valence_coarse);
      for (auto &elem : valence_coarse)
        elem = 1.0 / elem;

      // // define interapolation function and its transposed
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

      // TODO: const auto project_matrix = [&] (auto &dst, const &src) {}

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

      // // Specialization of projection for the case where src is the P0 basis
      // // function of a single cell Works only for P0 coarse elements
      // const auto project_cell = [&](auto &dst, const auto &cell) {
      //   AssertDimension(fe_coarse->n_dofs_per_cell(), 1);
      //   // TODO: vector problem requires dim, not 1!
      //   VectorType vec_local_coarse(fe_coarse->n_dofs_per_cell());
      //   VectorType vec_local_fine(fe_fine->n_dofs_per_cell());
      //   VectorType weights(fe_coarse->n_dofs_per_cell());

      //   const auto cell_coarse =
      //   cell->as_dof_handler_iterator(dh_coarse_patch); const auto cell_fine
      //   = cell->as_dof_handler_iterator(dh_fine_patch);

      //   // cell_coarse->get_dof_values(src, vec_local_coarse);
      //   vec_local_coarse[0] = 1.0;

      //   cell_coarse->get_dof_values(valence_coarse, weights);
      //   vec_local_coarse.scale(weights);

      //   projection_matrix.Tvmult(vec_local_fine, vec_local_coarse);

      //   cell_fine->distribute_local_to_global(vec_local_fine, dst);
      // };

      // we now compute c_loc_i = S^-1 P^T (P_tilda S^-1 P^T)^-1 e_i
      // where e_i is the indicator function of the patch

      VectorType P_e_i(Ndofs_fine);
      VectorType e_i(Ndofs_coarse); // reused also as temporary vector
      VectorType triple_product_inv_e_i(Ndofs_coarse);
      VectorType c_i(N_internal_dofs);
      VectorType Ac_i(N_internal_dofs);

      FullMatrix<double> PT(Ndofs_fine, Ndofs_coarse);
      FullMatrix<double> PT_internal(N_internal_dofs, Ndofs_coarse);
      FullMatrix<double> Ainv_PT(N_internal_dofs, Ndofs_coarse);
      // SparseMatrix<double>
      TrilinosWrappers::SparseMatrix
        internal_patch_stiffness_matrix(N_internal_dofs,N_internal_dofs, N_internal_dofs);
      FullMatrix<double> A_internal_full(N_internal_dofs, N_internal_dofs);
      // LAPACKFullMatrix<double>
      FullMatrix<double> P_Ainv_PT(Ndofs_coarse);


      computing_timer.leave_subsection();
      computing_timer.enter_subsection(
        "2: compute basis function 5a: projection");

      // assign rhs
      // TODO: projection that works on matrices!
      for (unsigned int i = 0; i < Ndofs_coarse; ++i)
        {
          e_i    = 0.0;
          P_e_i  = 0.0;
          e_i[i] = 1.0;

          if (true)
            project(P_e_i, e_i);
          else // assemble P_e_i as a fine rhs
            {
              //   const unsigned int dofs_per_cell =
              //   fe_fine->n_dofs_per_cell(); const unsigned int n_q_points =
              //   quadrature_fine->size();

              //   Vector<double> cell_rhs(dofs_per_cell);
              //   VectorType     rhs_values(1);

              //   FEValues<dim> fe_values(*fe_fine,
              //                           *quadrature_fine,
              //                           update_values | update_gradients |
              //                             update_quadrature_points |
              //                             update_JxW_values);

              //   for (const auto &cell :
              //        current_patch->sub_tria.active_cell_iterators())
              //     {
              //       const auto cell_coarse =
              //         cell->as_dof_handler_iterator(dh_coarse_patch);

              //       cell_coarse->get_dof_values(e_i, rhs_values);
              //       const auto cell_fine =
              //         cell->as_dof_handler_iterator(dh_fine_patch);
              //       cell_rhs = 0;
              //       fe_values.reinit(cell);
              //       for (unsigned int q = 0; q < n_q_points; ++q)
              //         {
              //           for (unsigned int i = 0; i < dofs_per_cell; ++i)
              //             {
              //               cell_rhs(i) += fe_values.shape_value(i, q) *
              //                              rhs_values[0] * fe_values.JxW(q);
              //             }
              //         }
              //       cell_fine->distribute_local_to_global(cell_rhs, P_e_i);
              //     }
            }

          for (unsigned int j = 0; j < Ndofs_fine; ++j)
            PT.set(j, i, P_e_i[j]);

          if (false)
            {
              dealii::TrilinosWrappers::SolverDirect sd(
                par.fine_solver_control);
              sd.solve(
                unconstrained_patch_stiffness_matrix,
                c_i,
                P_e_i); // wrong ! should be internal_patch_stiffness_matrix but
                        // its' not trilinos!
              // c_i = Ainv * P_e_i;
              e_i = 0.0;
              projectT(e_i, c_i);
              for (unsigned int j = 0; j < Ndofs_coarse; j++)
                {
                  P_Ainv_PT(j, i) = e_i[j];
                }
              for (unsigned int j = 0; j < Ndofs_fine; j++)
                {
                  Ainv_PT(j, i) = c_i[j];
                }
            } // substituted by gauss elimination
        }
computing_timer.leave_subsection();
      computing_timer.enter_subsection(
        "2: compute basis function 5b2: extraction 1");
        // std::cout << "A unconstrained" << std::endl;
        // unconstrained_patch_stiffness_matrix.print(std::cout);

//       A_internal_full.extract_submatrix_from(
//         unconstrained_patch_stiffness_matrix,
//         internal_dofs_fine,
//         internal_dofs_fine);

// // std::cout << "A full" << std::endl;
// //         A_internal_full.print(std::cout);

//         computing_timer.leave_subsection();
//       computing_timer.enter_subsection(
//         "2: compute basis function 5b2: extraction 2");
//       SparsityPattern isp;
//       // isp.copy_from(internal_sparsity_pattern);
//       isp.copy_from(A_internal_full);
//       computing_timer.leave_subsection();
//       computing_timer.enter_subsection(
//         "2: compute basis function 5b2: extraction 3");
//       internal_patch_stiffness_matrix.reinit(isp);
//       //internal_patch_stiffness_matrix.print(std::cout);
//       internal_patch_stiffness_matrix.copy_from(A_internal_full);

      
// // std::cout << "A int" << std::endl;
// //         internal_patch_stiffness_matrix.print(std::cout);

for (unsigned int i  = 0; i < N_internal_dofs; ++i)
for (unsigned int j  = 0; j < N_internal_dofs; ++j)
if (patch_sparsity_pattern.exists(internal_dofs_fine[i],internal_dofs_fine[j]))
{
internal_patch_stiffness_matrix.set(i, j, unconstrained_patch_stiffness_matrix(internal_dofs_fine[i],internal_dofs_fine[j]));
//std::cout << "(" << i << ", " << j << ") " << internal_patch_stiffness_matrix(i, j) << std::endl;
}
      


internal_patch_stiffness_matrix.compress(VectorOperation::insert);
// internal_patch_stiffness_matrix.print(std::cout);
      
      computing_timer.leave_subsection();
      computing_timer.enter_subsection(
        "2: compute basis function 5b2: extraction 4");

      PT_internal.extract_submatrix_from(PT,
                                         internal_dofs_fine,
                                         all_dofs_coarse);
      computing_timer.leave_subsection();
      computing_timer.enter_subsection(
        "2: compute basis function 5b: gauss_elimination");

      //       Gauss_elimination(PT_internal, patch_stiffness_matrix, Ainv_PT);
      Gauss_elimination(PT_internal, internal_patch_stiffness_matrix, Ainv_PT);

      computing_timer.leave_subsection();
      computing_timer.enter_subsection(
        "2: compute basis function 5c: triple product inversion");

      PT_internal.Tmmult(P_Ainv_PT, Ainv_PT);

      // P_tilda is actually P/ H^dim
      P_Ainv_PT /= (H * H); // H^

      P_Ainv_PT.gauss_jordan();

      computing_timer.leave_subsection();

      std::vector<VectorType> candidates;
      std::vector<VectorType> Palpha_i;
      VectorType              Pa_i(Ndofs_fine);
      Vector<double>          selected_basis_function(Ndofs_fine);
      Vector<double>          internal_selected_basis_function(N_internal_dofs);

      // FullMatrix<double> ID(Ndofs_coarse);
      // FullMatrix<double> temp(N_internal_dofs, Ndofs_coarse);
      // Ainv_PT.mmult(temp, P_Ainv_PT);
      // PT_internal.Tmmult(ID, temp);

      // ID /= (H*H);

      // ID.print_formatted(std::cout);


      if (!par.LOD_stabilization)
        {
          computing_timer.enter_subsection(
            "2: compute basis function 7: non stabilizaziona & assignemnt");
          // if we are not stabilizing then we only take the first candiadates
          // 0 is the index of the central cell
          // (this is also the central dof because we use P0 elements)

          e_i                    = 0.0;
          triple_product_inv_e_i = 0.0;

          e_i[0] = 1.0;
          P_Ainv_PT.vmult(triple_product_inv_e_i, e_i);

          Ainv_PT.vmult(internal_selected_basis_function,
                        triple_product_inv_e_i);
          // internal_selected_basis_function /=
          //   internal_selected_basis_function.l2_norm();

          computing_timer.leave_subsection();
        }
      else // SLOD
        {
          computing_timer.enter_subsection(
            "2: compute basis function 7: stabilizazion: setup");

          FullMatrix<double> S_boundary(N_boundary_dofs, N_internal_dofs);
          FullMatrix<double> PT_boundary(N_boundary_dofs, Ndofs_coarse);
          FullMatrix<double> BD(N_boundary_dofs, Ndofs_coarse);
          FullMatrix<double> B_full(N_boundary_dofs, Ndofs_coarse);
          // FullMatrix<double>       B(N_boundary_dofs, considered_candidates);
          // FullMatrix<double>       BDTBD(Ndofs_coarse, Ndofs_coarse);
          FullMatrix<double>       BDTBD(considered_candidates,
                                   considered_candidates);
          LAPACKFullMatrix<double> SVD(considered_candidates,
                                       considered_candidates);

          computing_timer.leave_subsection();
          // computing_timer.enter_subsection(
          //   "2: compute basis function 7b: stabilizazion:stiffness_assembly");
          // unconstrained_patch_stiffness_matrix.clear();
          // unconstrained_patch_stiffness_matrix.reinit(patch_sparsity_pattern);
          // AffineConstraints<double> empty_constraint;
          // empty_constraint.close();
          // LA::MPI::Vector dummy;
          // assemble_stiffness(unconstrained_patch_stiffness_matrix,
          //                    dummy,
          //                    dh_fine_patch,
          //                    empty_constraint);

          // computing_timer.leave_subsection();
          computing_timer.enter_subsection(
            "2: ompute basis function 7b: stabilizazion: extraction and othersetup");
          S_boundary.extract_submatrix_from(
            unconstrained_patch_stiffness_matrix,
            boundary_dofs_fine,
            internal_dofs_fine);

            
// std::cout << "S bd" << std::endl;
//         S_boundary.print(std::cout);
          // all_dofs_fine);
          //   if (false) // use sparse amtrix // mmutl does not work for spase
          //   with two full ones
          //   {
          //   SparsityPattern sp;
          //   sp.copy_from(S_boundary);
          //   SparseMatrix<double> S_boundary_sparse;
          //   S_boundary_sparse.reinit(sp);
          //   S_boundary_sparse.copy_from(S_boundary);

          //   S_boundary_sparse.mmult(B_full, Ainv_PT);
          // }
          // else

          // FullMatrix<double> Ainv_PT_restricted(N_internal_dofs,
          // Ndofs_coarse); Ainv_PT_restricted.extract_submatrix_from(Ainv_PT,
          //                                           internal_dofs_fine,
          //                                           all_dofs_coarse);
          S_boundary.mmult(B_full, Ainv_PT);
          PT_boundary.extract_submatrix_from(PT,
                                             boundary_dofs_fine,
                                             all_dofs_coarse);

          // PT_boundary.print(std::cout);
          PT_boundary *= -1;
          B_full.mmult(BD, P_Ainv_PT);

          PT_boundary.mmult(BD, P_Ainv_PT, true);

          // BD.Tmmult(BDTBD, BD);

          VectorType DeT(Ndofs_coarse);
          e_i    = 0.0;
          e_i[0] = 1.0;
          P_Ainv_PT.vmult(DeT, e_i);

          VectorType B_d0(N_boundary_dofs);

          std::vector<unsigned int> other_phi;
          for (unsigned int i = 0; i < considered_candidates; ++i)
            other_phi.push_back(i + 1);

          std::vector<unsigned int> temp;
          for (unsigned int i = 0; i < N_boundary_dofs; ++i)
            {
              B_d0[i] = BD(i, 0); // B_full(i, 0); // BD(i, 0);
              temp.push_back(i);
            }
          // B.extract_submatrix_from(B_full, temp, other_phi);

          VectorType d_i(considered_candidates);
          VectorType BDTBD0(considered_candidates);
          d_i    = 0;
          BDTBD0 = 0;

          FullMatrix<double> newBD(N_boundary_dofs, considered_candidates);

          newBD.extract_submatrix_from(BD, temp, other_phi);

          newBD.Tmmult(BDTBD, newBD);

          newBD.Tvmult(BDTBD0, B_d0);
          computing_timer.leave_subsection();
          computing_timer.enter_subsection(
            "compute basis function 7c: stabilizazion:correction and svd");

          SVD.copy_from(BDTBD);

          SVD.compute_inverse_svd(); // stores U V as normal, but
                                     // 1/singular_value_i
          d_i = 0.0;
          SVD.vmult(d_i, BDTBD0);
          d_i *= -1;

          // {
          //           SVD.compute_svd();
          //           auto               U  = SVD.get_svd_u();
          //           auto               Vt = SVD.get_svd_vt();
          //           FullMatrix<double> Sigma_minus1(considered_candidates);
          //           for (unsigned int i = 0; i < considered_candidates; ++i)
          //             Sigma_minus1(i, i) = (1 /
          //             SVD_manual.singular_value(i));
          //           d_i = 0;
          //           VectorType tt(considered_candidates);
          //           VectorType tt1(considered_candidates);
          //           U.Tvmult(tt, BDTBD0);
          //           Sigma_minus1.vmult(tt1, tt);
          //           Vt.Tvmult(d_i, tt1);
          //           d_i *=-1;
          // } // equivalent to previous (same d_i as output)

          auto U  = SVD.get_svd_u();
          auto Vt = SVD.get_svd_vt();

          AssertDimension(SVD.m(), SVD.n());
          AssertDimension(U.m(), U.n());
          AssertDimension(Vt.m(), Vt.n());
          AssertDimension(U.m(), Vt.n());
          AssertDimension(U.m(), SVD.n());
          AssertDimension(U.m(), considered_candidates);

          bool corrected = false;

          for (int i = (considered_candidates - 1); i >= 0; --i)
            {
              if (d_i.linfty_norm() < 0.5)
                break;
              corrected = true;
              VectorType uT(considered_candidates);
              VectorType v(considered_candidates);
              // for (auto j : all_dofs_coarse)
              for (unsigned int j = 0; j < considered_candidates; ++j)
                {
                  uT[j] = U(j, i);
                  v[j]  = Vt(i, j);
                }
              FullMatrix<double> vuT(considered_candidates,
                                     considered_candidates);
              vuT.outer_product(v, uT);
              VectorType correction(d_i.size());
              vuT.vmult(correction, BDTBD0);
              correction *= // Sigma_minus1(i, i); //
                SVD.singular_value(i);

              d_i += correction;
            }

          c_i = DeT;

          for (unsigned int index = 0; index < considered_candidates; ++index)
            {
              e_i            = 0.0;
              e_i[index + 1] = 1.0;

              P_Ainv_PT.vmult(DeT, e_i);

              c_i += d_i[index] * DeT;
            }

          Ainv_PT.vmult(internal_selected_basis_function, c_i);

          // internal_selected_basis_function /=
          //   internal_selected_basis_function.l2_norm();

          computing_timer.leave_subsection();
        }
      computing_timer.enter_subsection(
        "2: compute basis function 5d: xtend & asign");
      extend_vector_to_boundary_values(internal_selected_basis_function,
                                       dh_fine_patch,
                                       selected_basis_function);
      // std::cout << "internal" << std::endl;
      // internal_selected_basis_function.print(std::cout);
      // std::cout << "extended" << std::endl;
      // selected_basis_function.print(std::cout);

      current_patch->basis_function.push_back(selected_basis_function);
      Ac_i = 0;

      // unconstrained_patch_stiffness_matrix.vmult(Ac_i,
      // selected_basis_function); std::cout << "rhere 10" << std::endl;
      // TODO I think this should actually be
      VectorType Ac_i_0(Ndofs_fine);

      internal_patch_stiffness_matrix.vmult(Ac_i,
                                            internal_selected_basis_function);

      extend_vector_to_boundary_values(Ac_i, dh_fine_patch, Ac_i_0);

      current_patch->basis_function_premultiplied.push_back(Ac_i_0);

      dh_fine_patch.clear();

      computing_timer.leave_subsection();
    }
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::create_mesh_for_patch(Patch<dim> &current_patch)
{
  TimerOutput::Scope t(computing_timer, "1: Create mesh for Patches");

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

template <int dim, int spacedim>
void
LOD<dim, spacedim>::assemble_global_matrix()
{
  TimerOutput::Scope t(computing_timer, "3: Assemble global matrix");

  DoFHandler<dim> dh_fine_current_patch;

  // auto     lod = dh_fine_current_patch.locally_owned_dofs();
  // TODO: for mpi should not allocate all cols and rows->create partitioning
  // like we do for global_stiffness_matrix.reinit(..)

 //  basis_matrix.reinit(patches_pattern_fine.nonempty_rows(),
 //                      patches_pattern_fine.nonempty_cols(),
 //                      patches_pattern_fine,
 //                      mpi_communicator);

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

  basis_matrix_transposed.reinit(patches_pattern_fine_T.nonempty_rows(),
                                 patches_pattern_fine_T.nonempty_cols(),
                                 patches_pattern_fine_T,
                                 mpi_communicator);

  /*
  premultiplied_basis_matrix.reinit(
      patches_pattern_fine.nonempty_rows(),
      patches_pattern_fine.nonempty_cols(),
      patches_pattern_fine,
      mpi_communicator);
  */
  // basis_matrix               = 0.0;
  premultiplied_basis_matrix = 0.0;
  basis_matrix_transposed    = 0.0;


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
            current_patch->basis_function[0], phi_loc);
          AssertDimension(global_dofs.size(), phi_loc.size());
          // basis_matrix.set(current_patch_id,
          //                  phi_loc.size(),
          //                  global_dofs.data(),
          //                  phi_loc.data());
          for (unsigned int idx = 0; idx < phi_loc.size(); ++idx)
            {
              basis_matrix_transposed.set(global_dofs.data()[idx],
                                          current_patch_id,
                                          phi_loc.data()[idx]);
            }


          iterator_to_cell_in_current_patch->get_dof_values(
            current_patch->basis_function_premultiplied[0], phi_loc);
          AssertDimension(global_dofs.size(), phi_loc.size());
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
 //  basis_matrix.compress(VectorOperation::insert);
  premultiplied_basis_matrix.compress(VectorOperation::insert);
  basis_matrix_transposed.compress(VectorOperation::insert);

  // basis_matrix.mmult(global_stiffness_matrix, premultiplied_basis_matrix);

  //  global_stiffness_matrix.compress(VectorOperation::add);
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::assemble_stiffness( // Patch<dim> & current_patch,
  LA::MPI::SparseMatrix /*<double>*/ &stiffness_matrix,
  LA::MPI::Vector &                   rhs,
  const DoFHandler<dim> &             dh,
  AffineConstraints<double> &         stiffnes_constraints)
{
  stiffness_matrix = 0;
  if (rhs.size())
    rhs = 0;
  FEValues<dim> fe_values(*fe_fine,
                          *quadrature_fine,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe_fine->n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_fine->size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>                  rhs_values(n_q_points);

  const auto lexicographic_to_hierarchic_numbering =
    FETools::lexicographic_to_hierarchic_numbering<dim>(par.n_subdivisions);

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        if (rhs.size())
          par.rhs.value_list(fe_values.get_quadrature_points(), rhs_values);

        for (unsigned int c_1 = 0; c_1 < par.n_subdivisions; ++c_1)
          for (unsigned int c_0 = 0; c_0 < par.n_subdivisions; ++c_0)
            for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
              for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                {
                  const unsigned int q_index =
                    (c_0 * 2 + q_0) +
                    (c_1 * 2 + q_1) * (2 * par.n_subdivisions);

                  for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
                    for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
                      {
                        const unsigned int i =
                          lexicographic_to_hierarchic_numbering
                            [(c_0 + i_0) +
                             (c_1 + i_1) * (par.n_subdivisions + 1)];

                        for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                          for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                            {
                              const unsigned int j =
                                lexicographic_to_hierarchic_numbering
                                  [(c_0 + j_0) +
                                   (c_1 + j_1) * (par.n_subdivisions + 1)];

                              cell_matrix(i, j) +=
                                (fe_values.shape_grad(i, q_index) *
                                 fe_values.shape_grad(j, q_index) *
                                 fe_values.JxW(q_index));
                            }
                        if (rhs.size())
                          cell_rhs(i) += fe_values.shape_value(i, q_index) *
                                         rhs_values[q_index] *
                                         fe_values.JxW(q_index);
                      }
                }

        cell->get_dof_indices(local_dof_indices);
        if (rhs.size())
          stiffnes_constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, stiffness_matrix, rhs);
        else
          stiffnes_constraints.distribute_local_to_global(cell_matrix,
                                                          local_dof_indices,
                                                          stiffness_matrix);
      }
  stiffness_matrix.compress(VectorOperation::add);
  rhs.compress(VectorOperation::add);
}



template <int dim, int spacedim>
void
LOD<dim, spacedim>::solve()
{
  TimerOutput::Scope       t(computing_timer, "4: Solve LOD");
  LA::MPI::PreconditionAMG prec_A;
  prec_A.initialize(global_stiffness_matrix, 1.2);

  // const auto CT = linear_operator<LA::MPI::Vector>(basis_matrix_transposed);
  // const auto C  = transpose_operator(CT);
  // const auto AM =
  // linear_operator<LA::MPI::Vector>(premultiplied_basis_matrix);
  // const auto AMT = transpose_operator(AM);
  // const auto A = M*AMT;
  // const auto A    = linear_operator<LA::MPI::Vector>(global_stiffness_matrix);
  // auto       invA = A;
// 
  // const auto amgA = linear_operator(A, prec_A);

  SolverCG<LA::MPI::Vector> solver(par.coarse_solver_control);
  //invA = inverse_operator(A, solver, amgA);


  // system_rhs = C * fem_rhs;
  basis_matrix_transposed.Tvmult(system_rhs, fem_rhs);
  pcout << "     rhs l2 norm = " << system_rhs.l2_norm() << std::endl;

  // Some aliases
  // auto &      u = solution;
  // const auto &f = system_rhs;

  // dealii::TrilinosWrappers::SolverDirect sd(par.coarse_solver_control);
  // sd.solve(global_stiffness_matrix, solution, system_rhs, prec_A);

  //solution = invA * system_rhs;

  std::cout << global_stiffness_matrix.m() << " " << global_stiffness_matrix.n() << std::endl;
  std::cout << solution.size() << std::endl;
  std::cout << system_rhs.size() << std::endl;
  solver.solve(global_stiffness_matrix, solution, system_rhs, prec_A);
  pcout << "   size of u " << solution.size() << std::endl;
  coarse_boundary_constraints.distribute(solution);
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::solve_fem_problem() //_and_compare() // const
{
  // TimerOutput::Scope t(computing_timer, "4: assemble & Solve fine FEM");
computing_timer.enter_subsection("4: assemble & Solve fine FEM");
  const auto &dh = dof_handler_fine;

  auto     locally_owned_dofs = dh.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dh, locally_relevant_dofs);

  // create sparsity pattern fr global fine matrix
  AffineConstraints<double> fem_constraints(locally_relevant_dofs);
  // DoFTools::make_hanging_node_constraints(dh, fem_constraints); // not needed
  // with global refinemnt
  VectorTools::interpolate_boundary_values(dh, 0, par.bc, fem_constraints);
  fem_constraints.close();

  LA::MPI::SparseMatrix fem_stiffness_matrix;

  if (true)
    {
      DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
      DoFTools::make_sparsity_pattern(dh,
                                      sparsity_pattern,
                                      fem_constraints,
                                      false);
      SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                                 locally_owned_dofs,
                                                 mpi_communicator,
                                                 locally_relevant_dofs);



      fem_stiffness_matrix.reinit(locally_owned_dofs,
                                  locally_owned_dofs,
                                  sparsity_pattern,
                                  mpi_communicator);
    }
  else
    {
      DynamicSparsityPattern sparsity_pattern;

      std::vector<types::global_dof_index> dofs_on_this_cell;

      for (const auto &cell : dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            dofs_on_this_cell.resize(dofs_per_cell);
            cell->get_dof_indices(dofs_on_this_cell);

            fem_constraints.add_entries_local_to_global(
              dofs_on_this_cell,
              sparsity_pattern,
              true,
              bool_dof_mask); // keep constrained entries must be true
          }

      sparsity_pattern.compress();
      fem_stiffness_matrix.reinit(sparsity_pattern);
    }
  fem_rhs.reinit(locally_owned_dofs, mpi_communicator);
  fem_solution.reinit(locally_owned_dofs, mpi_communicator);

  LA::MPI::Vector locally_relevant_solution(locally_owned_dofs,
                                            locally_relevant_dofs,
                                            mpi_communicator);

  assemble_stiffness(fem_stiffness_matrix, fem_rhs, dh, fem_constraints);
//   IndexSet boundary_dofs_set =
//         DoFTools::extract_boundary_dofs(dh);
//   std::vector<unsigned int> boundary_dofs_fine;
//       boundary_dofs_set.fill_index_vector(boundary_dofs_fine);

//       const unsigned int N_boundary_dofs       = boundary_dofs_fine.size();
//       FullMatrix<double> A_boundary(N_boundary_dofs, N_boundary_dofs);
// A_boundary.extract_submatrix_from(fem_stiffness_matrix, boundary_dofs_fine, boundary_dofs_fine);
// A_boundary.print(std::cout);


  pcout << "     fem rhs l2 norm = " << fem_rhs.l2_norm() << std::endl;

  // solve
  LA::MPI::PreconditionAMG prec_Sh;
  prec_Sh.initialize(fem_stiffness_matrix, 1.2);
  //
  // const auto Sh    = linear_operator<LA::MPI::Vector>(fem_stiffness_matrix);
  // auto       invSh = Sh;
  //
  // const auto amg = linear_operator(Sh, prec_Sh);
  //
  // SolverCG<LA::MPI::Vector> cg_stiffness(par.fine_solver_control);
  // invSh = inverse_operator(Sh, cg_stiffness, amg);
  //
  // fem_solution = invSh * fem_rhs;

  SolverCG<LA::MPI::Vector> solver(par.fine_solver_control);
  solver.solve(fem_stiffness_matrix, fem_solution, fem_rhs, prec_Sh);

  pcout << "   size of fem u " << fem_solution.size() << std::endl;
  fem_constraints.distribute(fem_solution);
    computing_timer.leave_subsection();
  computing_timer.enter_subsection("4: solve LOD w extraction");


  // LA::MPI::SparseMatrix A_lod_temp;
  // LA::MPI::SparseMatrix A_lod;
  // std::cout << "basis_matrix " << basis_matrix.m() << " " << basis_matrix.n()
  // << std::endl; std::cout << "basis_matrix t " << basis_matrix_transposed.m()
  // << " " << basis_matrix_transposed.n() << std::endl; std::cout <<
  // "fem_stiffness_matrix " << fem_stiffness_matrix.m() << " " <<
  // fem_stiffness_matrix.n() << std::endl; basis_matrix.mmult(A_lod_temp,
  // fem_stiffness_matrix); std::cout << "A_lod_temp " << A_lod_temp.m() << " "
  // << A_lod_temp.n() << std::endl; A_lod_temp.mmult(A_lod,
  // basis_matrix_transposed);
  //
  // std::cout << "A_lod " << A_lod.m() << " " << A_lod.n() << " frobenius norm:
  // " << A_lod.frobenius_norm() << std::endl; std::cout <<
  // "global_stiffness_matrix " << global_stiffness_matrix.m() << " " <<
  // global_stiffness_matrix.n() << " frobenius norm: " <<
  // global_stiffness_matrix.frobenius_norm() << std::endl;

  // A_lod.print(std::cout);
  // global_stiffness_matrix.print(std::cout);

  /*
      auto Ndofs_coarse = tria.n_active_cells();
      auto Ndofs_fine   = dh.n_dofs();

  IndexSet boundary_dofs_set =
        DoFTools::extract_boundary_dofs(dh);

      std::vector<unsigned int> boundary_dofs_fine;
      boundary_dofs_set.fill_index_vector(boundary_dofs_fine);

      std::vector<unsigned int> internal_dofs_fine;
      std::vector<unsigned int> all_dofs_fine;
      // TODO : change, this is ugly
      for (unsigned int i = 0; i < Ndofs_fine; ++i)
        {
          all_dofs_fine.push_back(i);
          if (!boundary_dofs_set.is_element(i))
            internal_dofs_fine.push_back(i);
        }
      std::vector<unsigned int> all_dofs_coarse(all_dofs_fine.begin(),
                                                all_dofs_fine.begin() +
                                                  Ndofs_coarse);

      const unsigned int N_boundary_dofs       = boundary_dofs_fine.size();
      const unsigned int N_internal_dofs       = internal_dofs_fine.size();

  
  FullMatrix<double> BMT(N_internal_dofs, Ndofs_coarse);
  FullMatrix<double> FEMA(N_internal_dofs, N_internal_dofs);
  FullMatrix<double> ALT(N_internal_dofs, Ndofs_coarse);
  FullMatrix<double> GSM(Ndofs_coarse, Ndofs_coarse);
FEMA.extract_submatrix_from(fem_stiffness_matrix, internal_dofs_fine, internal_dofs_fine);
BMT.extract_submatrix_from(basis_matrix_transposed, internal_dofs_fine, all_dofs_coarse);
FEMA.mmult(ALT, BMT);
BMT.Tmmult(GSM, ALT);

// Vector<double> SR
Vector<double> FR(N_internal_dofs);
Vector<double> SR(Ndofs_coarse);
Vector<double> SS(Ndofs_coarse);
Assert(fem_rhs.size() == Ndofs_fine, ExcNotImplemented());
unsigned int id = 0;
for (unsigned int i = 0; i < Ndofs_fine; ++i)
{
  if (!boundary_dofs_set.is_element(i))
  {
    FR[id] = fem_rhs(i);
    id++;
  }
}

SolverCG<Vector<double>> sd(par.coarse_solver_control);
sd.solve(GSM,SS, SR, PreconditionIdentity());


  Vector<double> lod_solution(Ndofs_fine);
  Vector<double> FS(fem_solution);
Vector<double> LS(N_internal_dofs);
BMT.vmult(LS, SS);
extend_vector_to_boundary_values(LS, dh, lod_solution);

  par.convergence_table_compare.difference(dh, FS, lod_solution);
*/
      LA::MPI::SparseMatrix A_lod_temp;
  fem_stiffness_matrix.mmult(A_lod_temp, basis_matrix_transposed);
  basis_matrix_transposed.Tmmult(global_stiffness_matrix, A_lod_temp);
  // basis_matrix.mmult(A_lod_temp, fem_stiffness_matrix);
  // A_lod_temp.mmult(global_stiffness_matrix, basis_matrix_transposed);
  global_stiffness_matrix.compress(VectorOperation::add);

    computing_timer.leave_subsection();
/*
    auto Ndofs_coarse = tria.n_active_cells();
    auto Ndofs_fine   = dh.n_dofs();

      double H = pow(0.5, par.n_global_refinements);
  double h = H / (par.n_subdivisions);

  FullMatrix<double> projection_matrix(fe_coarse->n_dofs_per_cell(),
                                       fe_fine->n_dofs_per_cell());
  projection_P0_P1<dim>(projection_matrix);
  projection_matrix *= (h * h / 4);
       FullMatrix<double> PT(Ndofs_fine, Ndofs_coarse);
        FullMatrix<double> is_id(Ndofs_coarse, Ndofs_coarse);

Vector<double> valence_coarse(Ndofs_coarse);
      Vector<double> local_identity_coarse(fe_coarse->n_dofs_per_cell());
      local_identity_coarse = 1.0;

      for (const auto &cell : dof_handler_coarse.active_cell_iterators())
        cell->distribute_local_to_global(local_identity_coarse, valence_coarse);
      for (auto &elem : valence_coarse)
        elem = 1.0 / elem;

  const auto project = [&](auto &dst, const auto &src) {
        Vector<double> vec_local_coarse(fe_coarse->n_dofs_per_cell());
        Vector<double> vec_local_fine(fe_fine->n_dofs_per_cell());
        Vector<double> weights(fe_coarse->n_dofs_per_cell());

        for (const auto &cell : tria.active_cell_iterators())
          // should be locally_owned ?
          {
            const auto cell_coarse =
              cell->as_dof_handler_iterator(dof_handler_coarse);
            const auto cell_fine = cell->as_dof_handler_iterator(dof_handler_fine);

            cell_coarse->get_dof_values(src, vec_local_coarse);

            cell_coarse->get_dof_values(valence_coarse, weights);
            vec_local_coarse.scale(weights);

            projection_matrix.Tvmult(vec_local_fine, vec_local_coarse);

            cell_fine->distribute_local_to_global(vec_local_fine, dst);
          }
      };
Vector<double> P_e_i(Ndofs_fine);
      Vector<double> e_i(Ndofs_coarse);
  for (unsigned int i = 0; i < Ndofs_coarse; ++i)
        {
          e_i    = 0.0;
          P_e_i  = 0.0;
          e_i[i] = 1.0;

            project(P_e_i, e_i);

          for (unsigned int j = 0; j < Ndofs_fine; ++j)
            PT.set(j, i, P_e_i[j]);
        }

DynamicSparsityPattern identity(patches_pattern_fine.nonempty_rows());
  for (unsigned int i = 0; i < patches_pattern_fine.n_rows(); ++i)
    identity.add(i, i);
  DynamicSparsityPattern patches_pattern_fine_T;
  patches_pattern_fine_T.compute_Tmmult_pattern(patches_pattern_fine, identity);

IndexSet boundary_dofs_set =
        DoFTools::extract_boundary_dofs(dh);

      std::vector<unsigned int> boundary_dofs_fine;
      boundary_dofs_set.fill_index_vector(boundary_dofs_fine);

      std::vector<unsigned int> internal_dofs_fine;
      std::vector<unsigned int> all_dofs_fine;
      // TODO : change, this is ugly
      for (unsigned int i = 0; i < Ndofs_fine; ++i)
        {
          all_dofs_fine.push_back(i);
          if (!boundary_dofs_set.is_element(i))
            internal_dofs_fine.push_back(i);
        }
      std::vector<unsigned int> all_dofs_coarse(all_dofs_fine.begin(),
                                                all_dofs_fine.begin() +
                                                  Ndofs_coarse);

            const unsigned int N_boundary_dofs       = boundary_dofs_fine.size();
      const unsigned int N_internal_dofs       = internal_dofs_fine.size();

                                                  

      FullMatrix<double> PT_int(N_internal_dofs, Ndofs_coarse);
      PT_int.extract_submatrix_from(PT, internal_dofs_fine, all_dofs_coarse);
   //    PT_int.print(std::cout);
// std::cout << "here" << std::endl;
Assert(basis_matrix_transposed.m() == Ndofs_fine, ExcNotImplemented());
Assert(basis_matrix_transposed.n() == Ndofs_coarse, ExcNotImplemented());
FullMatrix<double> BMT(N_internal_dofs, Ndofs_coarse);
BMT.extract_submatrix_from(basis_matrix_transposed, internal_dofs_fine, all_dofs_coarse);
// BMT.print(std::cout);
// BMT.copy_from(basis_matrix_transposed);
        PT_int.Tmmult(is_id, BMT);
        is_id /= (H*H);
//is_id.print_formatted(std::cout, 3, true, 0, " ", 1., 0.01, ",");
*/

}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::compare_fem_lod()
{
  computing_timer.enter_subsection("5: compare FEM vs LOD");
  const auto &dh = dof_handler_fine;

  LA::MPI::Vector lod_solution(patches_pattern_fine.nonempty_cols(),
                               mpi_communicator);
  lod_solution = 0;
  // const auto C  = linear_operator<LA::MPI::Vector>(basis_matrix);
  // const auto CT = transpose_operator(C);
  // const auto CT = linear_operator<LA::MPI::Vector>(basis_matrix_transposed);
  // lod_solution  = CT * solution;
  basis_matrix_transposed.vmult(lod_solution, solution);
  par.convergence_table_compare.difference(dh, fem_solution, lod_solution);
  // par.convergence_table_FEM.error_from_exact(dh,
  //                                            fem_solution,
  //                                            par.exact_solution);
  // par.convergence_table_LOD.error_from_exact(dh,
  //                                            lod_solution,
  //                                            par.exact_solution);

  computing_timer.leave_subsection();
  computing_timer.enter_subsection("6: fine output");

  // output fem solution
  std::vector<std::string> fem_names(spacedim, "fem_solution");
  std::vector<std::string> exact_solution_names(spacedim,
                                                "exact_solution_fine");
  std::vector<std::string> lod_names(spacedim, "lod_solution_fine");

  auto exact_vec(fem_solution);
  VectorTools::interpolate(dh, par.exact_solution, exact_vec);
  // to be added for MPI
  // auto exact_vec_locally_relevant(locally_relevant_solution);
  // exact_vec_locally_relevant = exact_vec;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      spacedim, DataComponentInterpretation::component_is_scalar
      //        DataComponentInterpretation::component_is_part_of_vector
    );
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
  if(false)
  {
  for (unsigned int i = 0; i < basis_matrix_transposed.n(); ++i)
  {
    const std::string name = "c_" + std::to_string(i);
    std::vector<std::string> names(spacedim, name);

    auto e_i = solution;
    e_i = 0;
    e_i[i] = 1.0;

    auto c_i(lod_solution);
    c_i = 0.0;
    basis_matrix_transposed.vmult(c_i, e_i);

    //c_i.print(std::cout);

    data_out.add_data_vector(dh,
                           c_i,
                           names,
                           //  DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  }
  }
  data_out.build_patches();
  const std::string filename = par.output_name + "_fine.vtu";
  data_out.write_vtu_in_parallel(par.output_directory + "/" + filename,
                                 mpi_communicator);

  std::ofstream pvd_solutions(par.output_directory + "/" + par.output_name +
                              "_fine.pvd");


  computing_timer.leave_subsection();
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::initialize_patches()
{
  // TimerOutput::Scope t(computing_timer, "Initialize patches");
  create_patches();
  // MPI Barrier
  check_nested_patches();

  for (auto current_patch_id : locally_owned_patches)
    {
      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];
      create_mesh_for_patch(*current_patch);
    }

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

template <int dim, int spacedim>
void
LOD<dim, spacedim>::test()
{
  Triangulation<dim> square;

  GridGenerator::hyper_cube(square);
  square.refine_global(par.n_global_refinements);

  FE_DGQ<dim>      fe1(0);
  FE_Q_iso_Q1<dim> fe2(par.n_subdivisions);

  DoFHandler<dim> dof_handler_1(square);
  DoFHandler<dim> dof_handler_2(square);

  dof_handler_1.distribute_dofs(fe1);
  dof_handler_2.distribute_dofs(fe2);

  std::cout << "number active cells: " << square.n_active_cells() << std::endl;

  std::cout << "P0 number dofs per cell: " << fe1.n_dofs_per_cell()
            << std::endl;
  std::cout << "Q_iso number dofs per cell: " << fe2.n_dofs_per_cell()
            << std::endl;

  FullMatrix<double> projection_matrix(fe1.n_dofs_per_cell(),
                                       fe2.n_dofs_per_cell());
  FETools::get_projection_matrix(fe2, fe1, projection_matrix);

  projection_matrix.print(std::cout);

  projection_matrix = 0.0;
  projection_P0_P1<dim>(projection_matrix);
  projection_matrix.print(std::cout);
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::run()
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
      // pcout << "LOD vs exact solution (fine mesh)" << std::endl;
      // par.convergence_table_LOD.output_table(pcout.get_stream());
      // pcout << "FEM vs exact solution (fine mesh)" << std::endl;
      // par.convergence_table_FEM.output_table(pcout.get_stream());
      if (!par.LOD_stabilization)
        pcout << "LOD vs FEM (fine mesh)" << std::endl;
      else
        pcout << "SLOD vs FEM (fine mesh)" << std::endl;
      par.convergence_table_compare.output_table(pcout.get_stream());
    }

}


template class LOD<2, 1>;
