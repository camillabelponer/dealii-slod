#ifndef dealii_diffusion_h
#define dealii_diffusion_h

#include <LOD.h>



template <int dim, int spacedim>
class DiffusionProblem : public LOD<dim, spacedim>
{
public:
  DiffusionProblem(const LODParameters<dim, spacedim> &par)
    : LOD<dim, spacedim>(par){};

  typedef LOD<dim, spacedim> lod;


protected:
  Vector<double> alpha;

  virtual void
  create_random_problem_coefficients() override
  {
    TimerOutput::Scope t(lod::computing_timer, "1: create random coeff");

    Triangulation<dim> fine_tria;
    GridGenerator::hyper_cube(fine_tria);
    const unsigned int ref = (int)log2(lod::par.n_subdivisions);
    Assert(
      pow(2, ref) == lod::par.n_subdivisions,
      ExcNotImplemented(
        "for consistency, choose a number of subdivisions that's a power of 2 when asking for random coefficients"));
    fine_tria.refine_global((lod::par.n_global_refinements + ref));

    DoFHandler<dim> dh_fine(fine_tria);
    dh_fine.distribute_dofs(FE_Q<dim>(1));

    double       H                = pow(0.5, lod::par.n_global_refinements);
    double       h                = H / (lod::par.n_subdivisions);
    unsigned int N_cells_per_line = (int)1 / h;
    unsigned int N_fine_cells     = pow(N_cells_per_line, dim);
    alpha.reinit(N_fine_cells);

    if (!lod::par.constant_coefficients)
      {
        // cells are on all processors (parallel::shared triangulation)
        // but they are on a 1 to 1 correspondence with the patches
        // we want to create a different coefficient for any fine cell !
        // we can use the same logic as used to create the patches but with
        // small h
        for (const auto &cell : lod::dof_handler_coarse.active_cell_iterators())
          // if locally owned but on the patch!!
          {
            // coordinates of the bottom left corner of the coarse cell
            const double x0 = cell->barycenter()(0) - H / 2;
            const double y0 = cell->barycenter()(1) - H / 2;

            double x = x0 + h / 2;
            while (x < (x0 + H))
              {
                double y = y0 + h / 2;
                while (y < (y0 + H))
                  {
                    const unsigned int vector_cell_index =
                      (int)floor(x / h) + N_cells_per_line * (int)floor(y / h);

                    alpha[vector_cell_index] =
                      1.0 + static_cast<float>(rand()) /
                              (static_cast<float>(RAND_MAX / (100.0 - 1.0)));
                    y += h;
                  }
                x += h;
              }
          }

        lod::data_out.attach_dof_handler(dh_fine);

        lod::data_out.add_data_vector(alpha,
                                      "alpha",
                                      DataOut<dim>::type_cell_data,
                                      lod::data_component_interpretation);

        lod::data_out.build_patches();
        const std::string filename = lod::par.output_name + "_coefficients.vtu";
        lod::data_out.write_vtu_in_parallel(lod::par.output_directory + "/" +
                                              filename,
                                            lod::mpi_communicator);

        // std::ofstream pvd_solutions(lod::par.output_directory + "/" +
        // filename +
        //                             "_fine.pvd");

        lod::data_out.clear();
      }
    else
      {
        for (unsigned int i = 0; i < N_fine_cells; ++i)
          {
            alpha[i] = 1.0;
          }
      }
  }


  virtual void
  assemble_stiffness(LA::MPI::SparseMatrix &    stiffness_matrix,
                     LA::MPI::Vector &          rhs,
                     const DoFHandler<dim> &    dh,
                     AffineConstraints<double> &stiffness_constraints) override
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
      FETools::lexicographic_to_hierarchic_numbering<dim>(
        lod::par.n_subdivisions);

    double       H                = pow(0.5, lod::par.n_global_refinements);
    double       h                = H / (lod::par.n_subdivisions);
    unsigned int N_cells_per_line = (int)1 / h;

    for (const auto &cell : dh.active_cell_iterators())
      // if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        const double       x = cell->barycenter()(0);
        const double       y = cell->barycenter()(1);
        const unsigned int vector_cell_index =
          (int)floor(x / h) + N_cells_per_line * (int)floor(y / h);

        if (rhs.size())
          {
            lod::par.rhs.value_list(fe_values.get_quadrature_points(),
                                    rhs_values);
          }

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
                                alpha[vector_cell_index] *
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
          stiffness_constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, stiffness_matrix, rhs);
        else
          stiffness_constraints.distribute_local_to_global(cell_matrix,
                                                           local_dof_indices,
                                                           stiffness_matrix);
      }
    stiffness_matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  };
};

#endif