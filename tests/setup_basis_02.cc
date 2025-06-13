// Compute and plot a basis.

#include <deal.II/base/function_signed_distance.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

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


  hp::FECollection<dim> fe_q1;
  fe_q1.push_back(FE_Q<dim>(1));
  QGauss<1> quadrature_q1_1D(2);

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, 1 + 2 * n_oversampling);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  compute_renumbering_lex(dof_handler);

  Triangulation<dim> tria_q1;
  GridGenerator::subdivided_hyper_cube(tria_q1,
                                       (1 + 2 * n_oversampling) *
                                         n_subdivisions_fine);
  DoFHandler<dim> dof_handler_q1(tria_q1);
  dof_handler_q1.distribute_dofs(fe_q1);
  compute_renumbering_lex(dof_handler_q1);

  AffineConstraints<double> constraints;
  constraints.close();

  // level set and classify cells
  const FE_Q<dim> fe_level_set(3);
  DoFHandler<dim> level_set_dof_handler(tria_q1);
  level_set_dof_handler.distribute_dofs(fe_level_set);

  Vector<double> level_set;
  level_set.reinit(level_set_dof_handler.n_dofs());

  NonMatching::MeshClassifier<dim> mesh_classifier(level_set_dof_handler,
                                                   level_set);

  const Functions::SignedDistance::Sphere<dim> signed_distance_sphere({}, 0.6);
  VectorTools::interpolate(level_set_dof_handler,
                           signed_distance_sphere,
                           level_set);

  mesh_classifier.reclassify();

  const auto face_has_ghost_penalty = [&](const auto        &cell,
                                          const unsigned int face_index) {
    if (cell->at_boundary(face_index))
      return false;

    const NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);

    const NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::outside)
      return true;

    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::outside)
      return true;

    return false;
  };

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

  if (true)
    {
      const auto face_has_flux_coupling = [&](const auto        &cell,
                                              const unsigned int face_index) {
        return face_has_ghost_penalty(cell, face_index);
      };

      Table<2, DoFTools::Coupling> cell_coupling(1, 1);
      Table<2, DoFTools::Coupling> face_coupling(1, 1);
      cell_coupling[0][0] = DoFTools::always;
      face_coupling[0][0] = DoFTools::always;

      DoFTools::make_flux_sparsity_pattern(dof_handler_q1,
                                           sparsity_pattern,
                                           constraints,
                                           true,
                                           cell_coupling,
                                           face_coupling,
                                           numbers::invalid_subdomain_id,
                                           face_has_flux_coupling);
    }
  else
    patch.create_sparsity_pattern(constraints, sparsity_pattern);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix patch_stiffness_matrix(sparsity_pattern);

  NonMatching::RegionUpdateFlags region_update_flags;
  region_update_flags.inside = update_values | update_gradients |
                               update_JxW_values | update_quadrature_points;
  region_update_flags.surface = update_values | update_gradients |
                                update_JxW_values | update_quadrature_points |
                                update_normal_vectors;

  NonMatching::FEValues<dim> non_matching_fe_values(fe_q1,
                                                    quadrature_q1_1D,
                                                    region_update_flags,
                                                    mesh_classifier,
                                                    level_set_dof_handler,
                                                    level_set);

  FEInterfaceValues<dim> fe_interface_values(
    fe_q1,
    hp::QCollection<dim - 1>(QGauss<dim - 1>(2)),
    update_gradients | update_JxW_values | update_normal_vectors);

  for (const auto &cell : dof_handler_q1.active_cell_iterators())
    {
      if (mesh_classifier.location_to_level_set(cell) ==
          NonMatching::LocationToLevelSet::outside)
        continue;

      non_matching_fe_values.reinit(cell);

      const double       cell_side_length  = cell->minimum_vertex_distance();
      const unsigned int fe_degree         = 1;
      const double       nitsche_parameter = 5 * (fe_degree + 1) * fe_degree;
      const double       ghost_parameter   = 0.5;

      const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

      const std::optional<FEValues<dim>> &inside_fe_values =
        non_matching_fe_values.get_inside_fe_values();

      if (inside_fe_values)
        {
          for (const auto q : inside_fe_values->quadrature_point_indices())
            for (const auto i : inside_fe_values->dof_indices())
              for (const auto j : inside_fe_values->dof_indices())
                cell_matrix(i, j) += (inside_fe_values->shape_grad(i, q) *
                                      inside_fe_values->shape_grad(j, q) *
                                      inside_fe_values->JxW(q));
        }



      const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
        &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

      if (surface_fe_values)
        {
          for (const unsigned int q :
               surface_fe_values->quadrature_point_indices())
            {
              const Tensor<1, dim> &normal =
                surface_fe_values->normal_vector(q);
              for (const unsigned int i : surface_fe_values->dof_indices())
                {
                  for (const unsigned int j : surface_fe_values->dof_indices())
                    {
                      cell_matrix(i, j) +=
                        (-normal * surface_fe_values->shape_grad(i, q) *
                           surface_fe_values->shape_value(j, q) +
                         -normal * surface_fe_values->shape_grad(j, q) *
                           surface_fe_values->shape_value(i, q) +
                         nitsche_parameter / cell_side_length *
                           surface_fe_values->shape_value(i, q) *
                           surface_fe_values->shape_value(j, q)) *
                        surface_fe_values->JxW(q);
                    }
                }
            }
        }


      for (const unsigned int f : cell->face_indices())
        if (face_has_ghost_penalty(cell, f))
          {
            fe_interface_values.reinit(cell,
                                       f,
                                       numbers::invalid_unsigned_int,
                                       cell->neighbor(f),
                                       cell->neighbor_of_neighbor(f),
                                       numbers::invalid_unsigned_int);

            const unsigned int n_interface_dofs =
              fe_interface_values.n_current_interface_dofs();
            FullMatrix<double> local_stabilization(n_interface_dofs,
                                                   n_interface_dofs);
            for (unsigned int q = 0;
                 q < fe_interface_values.n_quadrature_points;
                 ++q)
              {
                const Tensor<1, dim> normal = fe_interface_values.normal(q);
                for (unsigned int i = 0; i < n_interface_dofs; ++i)
                  for (unsigned int j = 0; j < n_interface_dofs; ++j)
                    {
                      local_stabilization(i, j) +=
                        .5 * ghost_parameter * cell_side_length * normal *
                        fe_interface_values.jump_in_shape_gradients(i, q) *
                        normal *
                        fe_interface_values.jump_in_shape_gradients(j, q) *
                        fe_interface_values.JxW(q);
                    }
              }

            const std::vector<types::global_dof_index>
              local_interface_dof_indices =
                fe_interface_values.get_interface_dof_indices();

            patch_stiffness_matrix.add(local_interface_dof_indices,
                                       local_stabilization);
          }

      std::vector<types::global_dof_index> indices(dofs_per_cell);
      cell->get_dof_indices(indices);

      constraints.distribute_local_to_global(cell_matrix,
                                             indices,
                                             patch_stiffness_matrix);
    }
  patch_stiffness_matrix.compress(VectorOperation::values::add);

  for (auto &entry : patch_stiffness_matrix)
    if ((entry.row() == entry.column()) && (entry.value() == 0.0))
      entry.value() = 1.0;

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

  if (true)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.add_data_vector(dof_handler,
                               selected_basis_function[0],
                               "basis");

      data_out.build_patches(mapping, n_subdivisions_fine);

      data_out.write_vtu_in_parallel("selected_basis_" + std::to_string(dim) +
                                       ".vtu",
                                     tria.get_communicator());
    }

  if (true)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.add_data_vector(level_set_dof_handler, level_set, "ls");

      data_out.build_patches(mapping, 3);

      data_out.write_vtu_in_parallel("ls_" + std::to_string(dim) + ".vtu",
                                     tria.get_communicator());
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<1>();
  test<2>();
}