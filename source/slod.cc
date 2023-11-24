#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constrained_linear_operator.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <slod.h>

template <int dim>
SLOD<dim>::SLOD(Triangulation<dim> tria)
  : tria(tria)
  , dof_handler(tria)
{}

template <int dim>
void
SLOD<dim>::make_fe()
{
  fe_coarse = std::make_unique(FE_DGQ<dim>(0));
  fe_fine   = std::make_unique(FE_Q_iso_Q1<dim>(n_subdivisions));
  dof_handler.distribute_dofs(fe_coarse);
  // TODO: 2?
  quadrature_fine = QIterated<dim>(QGauss<1>(2), n_subdivisions);
}

template <int dim>
void
SLOD<dim>::create_patches()
{
  // Queue for patches for which neighbours should be added
  std::vector<typename DoFHandler<dim>::active_cell_iterator> patch_iterators;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> dof_indices(
        fe_coarse.n_dofs_per_cell());
      cell->get_dof_indices(dof_indices);
      // cell_dof_indices[cell->active_cell_index()] = dof_indices;

      bool       crosses_border = false;
      Patch<dim> patch;
      patch_iterators.clear();
      patch_iterators.push_back(cell);
      // The iterators for level l are in the range [l_start, l_end) of
      // patch_iterators
      unsigned int l_start = 0;
      unsigned int l_end   = 1;
      patch.cells.add_index(cell->active_cell_index());
      for (unsigned int l = 1; l <= oversampling; l++)
        {
          for (unsigned int i = l_start; i <= l_end; i++)
            {
              if (patch_iterators[i].at_boundary())
                {
                  crosses_border = true;
                }
              for (auto vertex : patch_iterators[i].vertex_iterator())
                {
                  for (const auto &neighbour :
                       GridTools::find_cells_adjacent_to_vertex(tria, vertex))
                    {
                      if (!patch.cells.is_element(
                            neighbour->active_cell_index()))
                        {
                          patch_iterators.push_back(neighbour);
                        }
                      patch.cells.add_index(neighbour->active_cell_index());
                    }
                }
            }
          l_start = l_end;
          l_end   = patch_iterators.size();
        }
      //     if (crosses_border) {
      //       patch.num_basis_vectors = 0;
      //     } else {
      //       patch.num_basis_vectors = 1;
      //     }
      patches[cell->active_cell_index()] = patch;
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

template<int dim>
PatchXSq<dim>::PatchXSq() {}

template<int dim>
void PatchXSq<dim>::reinit(LinearOperator<Vector<double>> linop, DoFHandler<dim> &dh_coarse, DoFHandler<dim> &dh_fine) {
  reinit(dh_fine.n_dofs());
  op = linop;
  transfer.reinit_polynomial_transfer(dh_fine, dh_coarse);
  intermediate_fine.reinit(dh_fine.n_dofs(), true);
  intermediate_coarse.reinit(dh_coarse.n_dofs(), true);
}

template<int dim>
void PatchXSq<dim>::vmult_add(PETScWrappers::VectorBase &dst, PETScWrappers::VectorBase &src) {
  for (unsigned int i = 0; i < intermediate_fine.size(); i++) {
    intermediate_fine[i] = src[i];
  }
  intermediate_fine = op * intermediate_fine;
  transfer.restrict_and_add(intermediate_coarse, intermediate_fine);
  transfer.prolongate_and_add(intermediate_fine, intermediate_coarse);
  intermediate_fine = transpose_operator(op) * intermediate_fine;
  for (unsigned int i = 0; i < intermediate_coarse.size(); i++) {
    dst[i] += intermediate_coarse[i];
  }
}

template<int dim>
void PatchXSq<dim>::Tvmult_add(PETScWrappers::VectorBase &dst, PETScWrappers::VectorBase &src) {
  vmult_add(dst, src);
}

template<int dim>
void PatchXSq<dim>::vmult(PETScWrappers::VectorBase &dst, PETScWrappers::VectorBase &src) {
  dst = 0;
  vmult_add(dst, src);
}

template<int dim>
void PatchXSq<dim>::Tvmult(PETScWrappers::VectorBase &dst, PETScWrappers::VectorBase &src) {
  dst = 0;
  Tvmult_add(dst, src);
}

const unsigned int SPECIAL_NUMBER = 69;

template <int dim>
void
SLOD<dim>::compute_basis_function_candidates()
{
  DoFHandler<dim>           dh_coarse_patch;

  // TODO: reinit in loop
  FullMatrix<double>        patch_stiffness_matrix;
  AffineConstraints<double>        internal_boundary_constraints;
  PatchXSq<dim> patch_X_sq;

  std::vector<std::vector<double>> right_hand_sides;
  std::vector<double>              eigenvalues;

  SolverControl                solver_control(100, 1e-9);
  SLEPcWrappers::SolverLanczos eig_solver(solver_control);
  eig_solver.set_problem_type(EPS_HEP);
  eig_solver.set_which_eigenpairs(EPS_LARGEST_MAGNITUDE);

  const auto locally_owned_patches =
    Utilities::MPI::create_evenly_distributed_partitioning(mpi_communicator,
                                                           patches.size());
  // for (unsigned int patch_id = 0; patch_id < patches.size(); patch_id++) {
  for (auto &current_patch : locally_owned_patches)
    {
      //   if (patches[i].num_basis_vectors == 0) continue;

      create_mesh_for_patch(&current_patch, &current_patch.sub_tria);
      current_patch.dh.reinit(current_patch.sub_tria);
      current_patch.dh.distribute_dofs(*fe_fine);
      dh_coarse_patch.distribute_dofs(*fe_coarse);

      // TODO: Can fe_fine be used for different patches at the same time?
      assemble_stiffness_for_patch(&current_patch, patch_stiffness_matrix);
      VectorTools::interpolate_boundary_values(current_patch.dh, SPECIAL_NUMBER, 0, internal_boundary_constraints);

      const auto A = linear_operator(patch_stiffness_matrix);
      const auto A0 = constrained_linear_operator(internal_boundary_constraints, A);

      // Following the method of contrained_linear_operator, construct C^T * A * Id_c
      const auto C = distribute_constraints_linear_operator(internal_boundary_constraints, A);
      const auto Id_c = project_to_constrained_linear_operator(internal_boundary_constraints, A);
      const auto A1 = transpose_operator(C) * A * Id_c;

      auto A0_inv = A0;
      // TODO: preconditioning
      // TODO: solver contro
      SolverCG cg(solver_control);
      A0_inv = inverse_operator(A0, cg);

      const auto op = -1 * A0_inv * A1;

      patch_X_sq.reinit(op, dh_coarse_patch, current_patch.dh);

      right_hand_sides.clear();
      right_hand_sides.push_back(
        std::vector<double>(dh_coarse_patch.n_dofs(), 0.0));
      eig_solver.solve(patch_X_sq,
                       eigenvalues,
                       right_hand_sides,
                       num_basis_vectors);

      for (unsigned int j = 0; j < num_basis_vectors; j++)
        {
          Vector<double> right_hand_side_fine;
          patch_X_sq.transfer.prolongate(right_hand_side_fine, right_hand_sides[j]);
          right_hand_side_fine = A0_inv * right_hand_side_fine;
          current_patch.basis_function_candidates.push_back(right_hand_side_fine);
        }
    }
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
  std::vector<unsigned int> new_vertex_indices(tria.n_vertices(),
                                               numbers::invalid_unsigned_int);

  unsigned int c = 0;
  for (const auto &index : current_patch.cells)
    {
      auto cell = tria.create_cell_iterator(index);
      if (cell != tria.end())
        for (const unsigned int v : cell->vertex_indices())
          new_vertex_indices[cell->vertex_index(v)] = c++;
    }

  // collect points
  std::vector<Point<dim>> sub_points;
  for (unsigned int i = 0; i < new_vertex_indices.size(); ++i)
    if (new_vertex_indices[i] != numbers::invalid_unsigned_int)
      sub_points.emplace_back(tria.get_vertices()[i]);

  // create new cell and subcell data
  std::vector<CellData<dim>> sub_cells;

  for (const auto &index : current_patch.cells)
    {
      auto cell = tria.create_cell_iterator(index);
      if (cell != tria.end())
        {
          // cell
          CellData<dim> new_cell(cell->n_vertices());

          for (const auto v : cell->vertex_indices())
            new_cell.vertices[v] = new_vertex_indices[cell->vertex_index(v)];

          new_cell.material_id = cell->material_id();
          new_cell.manifold_id = cell->manifold_id();

          sub_cells.emplace_back(new_cell);
        }
    }

  // create mesh
  current_patch.sub_tria.create_triangulation(sub_points, sub_cells, {});

  auto sub_cell = current_patch.sub_tria.begin();

  for (const auto &index : current_patch.cells)
    {
      auto cell = tria.create_cell_iterator(index);
      if (cell != tria.end())
        {
          // faces
          for (const auto f : cell->face_indices())
            {
              const auto face = cell->face(f);

              if (face->manifold_id() != numbers::flat_manifold_id)
                sub_cell->face(f)->set_manifold_id(face->manifold_id());

              if (face->boundary_id() != numbers::internal_face_boundary_id)
                sub_cell->face(f)->set_boundary_id(face->boundary_id());
              else if (sub_cell->face(f)->boundary_id() !=
                       numbers::internal_face_boundary_id)
                sub_cell->face(f)->set_boundary_id(SPECIAL_NUMBER);
            }

          // lines
          if (dim == 3)
            for (const auto l : cell->line_indices())
              {
                const auto line = cell->line(l);

                if (line->manifold_id() != numbers::flat_manifold_id)
                  sub_cell->line(l)->set_manifold_id(line->manifold_id());
              }

          sub_cell++;
        }
    }
}

template <int dim>
void
SLOD<dim>::assemble_stiffness_for_patch(Patch<dim> &        current_patch,
                                        FullMatrix<double> &stiffness_matrix)
{
  const auto &dh = current_patch.dh_fine;
  // TODO: stiffness_matrix should be sparse
  // TODO: avoid reallocations
  // TODO
  // TimerOutput::Scope t(computing_timer, "Assemble Stiffness and Neumann
  // rhs");
  FEValues<dim> fe_values(*fe_fine,
                          *quadrature_fine,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int          dofs_per_cell = fe_fine->n_dofs_per_cell();
  const unsigned int          n_q_points    = quadrature_fine->size();
  FullMatrix<double>          cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  AffineConstraints<double>            constraints;

  // TODO: Parameter adaptable parameters
  VectorTools::interpolate_boundary_values(dh, 0, 0, constraints);

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
                grad_phi_u[k] = fe_values.gradient(k, q);
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
        // TODO: Handle inhomogeneous dirichlet bc
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               stiffness_matrix);
      }
}

