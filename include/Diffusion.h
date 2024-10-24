#ifndef dealii_diffusion_h
#define dealii_diffusion_h

#include <LOD.h>



template <int dim, int spacedim>
class DiffusionProblem : public LOD<dim, spacedim>
{
public:
  DiffusionProblem(const LODParameters<dim, spacedim> &par)
    : LOD<dim, spacedim>(par)
    {};

  typedef LOD<dim, spacedim> lod;


protected:

  virtual void
  output_coarse_results()
  {

  TimerOutput::Scope t(lod::computing_timer, "6: Output results");

  std::vector<std::string> solution_names(spacedim, "LOD_solution");
  std::vector<std::string> exact_solution_names(spacedim,
                                                "coarse_exact_solution");

  auto exact_vec(lod::solution);
  VectorTools::interpolate(lod::dof_handler_coarse, lod::par.exact_solution, exact_vec);
  // to be added for MPI
  // auto exact_vec_locally_relevant(locally_relevant_solution);
  // exact_vec_locally_relevant = exact_vec;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      spacedim,
      DataComponentInterpretation::component_is_part_of_vector);
  DataOut<dim> data_out;

  // data_out.attach_dof_handler(dof_handler_coarse);

  // data_out.add_data_vector(solution, solution_names);
  data_out.add_data_vector(lod::dof_handler_coarse,
                           lod::solution,
                           solution_names,
                           // DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.add_data_vector(lod::dof_handler_coarse,
                           exact_vec,
                           exact_solution_names,
                           //  DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  // Vector<float> subdomain(tria.n_active_cells());
  // for (unsigned int i = 0; i < subdomain.size(); ++i)
  //   subdomain(i) = tria.locally_owned_subdomain();
  // data_out.add_data_vector(subdomain, "subdomain");
  data_out.build_patches();
  const std::string filename = lod::par.output_name + ".vtu";
  data_out.write_vtu_in_parallel(lod::par.output_directory + "/" + filename,
                                 lod::mpi_communicator);

  std::ofstream pvd_solutions(lod::par.output_directory + "/" + lod::par.output_name +
                              ".pvd");
  };
  virtual void
  output_fine_results() {};

  virtual void
  assemble_stiffness(LA::MPI::SparseMatrix /*<double>*/ &stiffness_matrix,
                     LA::MPI::Vector &                   rhs,
                     const DoFHandler<dim> &             dh,
                     AffineConstraints<double> &         stiffnes_constraints)
  {
  stiffness_matrix = 0;
  if (rhs.size())
    rhs = 0;
  FEValues<dim> fe_values(*lod::fe_fine,
                          *lod::quadrature_fine,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = lod::fe_fine->n_dofs_per_cell();
  const unsigned int n_q_points    = lod::quadrature_fine->size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>                  rhs_values(n_q_points);

  const auto lexicographic_to_hierarchic_numbering =
    FETools::lexicographic_to_hierarchic_numbering<dim>(lod::par.n_subdivisions);

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        if (rhs.size())
          lod::par.rhs.value_list(fe_values.get_quadrature_points(), rhs_values);

        for (unsigned int c_1 = 0; c_1 < lod::par.n_subdivisions; ++c_1)
          for (unsigned int c_0 = 0; c_0 < lod::par.n_subdivisions; ++c_0)
            for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
              for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                {
                  const unsigned int q_index =
                    (c_0 * 2 + q_0) +
                    (c_1 * 2 + q_1) * (2 * lod::par.n_subdivisions);

                  for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
                    for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
                      {
                        const unsigned int i =
                          lexicographic_to_hierarchic_numbering
                            [(c_0 + i_0) +
                             (c_1 + i_1) * (lod::par.n_subdivisions + 1)];

                        for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                          for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                            {
                              const unsigned int j =
                                lexicographic_to_hierarchic_numbering
                                  [(c_0 + j_0) +
                                   (c_1 + j_1) * (lod::par.n_subdivisions + 1)];

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

  };

  virtual void
  assemble_stiffness_patch(SparseMatrix<double> & stiffness_matrix,
                           const DoFHandler<dim> &dh)  {};
};



#endif