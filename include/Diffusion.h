#ifndef dealii_diffusion_h
#define dealii_diffusion_h

#include <LOD.h>


template <int dim>
class problem_parameter : public Function<dim, double>
{
private:
  const double        min_val;
  const double        max_val;
  const unsigned int  refinement;
  std::vector<double> random_values;
  unsigned int        N_cells_per_line;
  double              eta;

public:
  problem_parameter(double min, double max, unsigned int r)
    : min_val(min)
    , max_val(max)
    , refinement(r)
  {
    N_cells_per_line     = pow(2, refinement);
    eta                  = (double)1 / N_cells_per_line;
    unsigned int N_cells = pow(N_cells_per_line, dim);
    // random_values.reinit(N_cells);
    if (max_val != min_val)
      {
        for (unsigned int i = 0; i < N_cells; ++i)
          {
            const double v =
              min_val + static_cast<float>(rand()) /
                          (static_cast<float>(RAND_MAX / (max_val - min_val)));
            random_values.push_back(v);
          }
      }
  };

  double
  value(const Point<dim> &p, const unsigned int) const override
  {
    if (max_val == min_val) // constant coefficients
      return min_val;
    else
      {
        const double x = p(0);
        const double y = p(1);
        unsigned int vector_cell_index =
          (int)floor(x / eta) + N_cells_per_line * (int)floor(y / eta);
        return random_values[vector_cell_index];
      }
  }
};

template <int dim, int spacedim>
class DiffusionProblem : public LOD<dim, spacedim>
{
public:
  DiffusionProblem(const LODParameters<dim, spacedim> &par)
    : LOD<dim, spacedim>(par)
    , Alpha(1, 1, 8){};

  typedef LOD<dim, spacedim> lod;


protected:
  problem_parameter<dim> Alpha;

  virtual void
  create_random_problem_coefficients() override
  {
    // TimerOutput::Scope t(lod::computing_timer, "1: create random coeff");

    Triangulation<dim> triangulation_h;
    GridGenerator::hyper_cube(triangulation_h);
    const unsigned int ref = (int)log2(lod::par.n_subdivisions);
    Assert(
      pow(2, ref) == lod::par.n_subdivisions,
      ExcNotImplemented(
        "for consistency, choose a number of subdivisions that's a power of 2"));
    triangulation_h.refine_global((lod::par.n_global_refinements + ref));

    DoFHandler<dim> dh_h(triangulation_h);
    dh_h.distribute_dofs(FE_DGQ<dim>(0)); //(FE_Q<dim>(1), spacedim);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      scalar_component_interpretation(
        1, DataComponentInterpretation::component_is_scalar);
    Vector<double> alpha_values;
    alpha_values.reinit(dh_h.n_dofs());

    VectorTools::interpolate(dh_h, Alpha, alpha_values);

    lod::data_out.attach_dof_handler(dh_h);
    lod::data_out.add_data_vector(alpha_values,
                                  "alpha",
                                  DataOut<dim>::type_dof_data,
                                  scalar_component_interpretation);

    lod::data_out.build_patches();
    const std::string filename = lod::par.output_name + "_coefficients.vtu";
    lod::data_out.write_vtu_in_parallel(lod::par.output_directory + "/" +
                                          filename,
                                        lod::mpi_communicator);

    lod::data_out.clear();
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
    std::vector<double>                  alpha_values(n_q_points);
    // using n_q_points gives us a vectore much larger than what we need due to
    // the Q_iso_Q1 nature

    const auto lexicographic_to_hierarchic_numbering =
      FETools::lexicographic_to_hierarchic_numbering<dim>(
        lod::par.n_subdivisions);


    for (const auto &cell : dh.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        if (rhs.size())
          {
            lod::par.rhs.value_list(fe_values.get_quadrature_points(),
                                    rhs_values);
          }
        Alpha.value_list(fe_values.get_quadrature_points(), alpha_values);

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
                                // alpha[vector_cell_index] *
                                alpha_values[q_index] *
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


  virtual void
  assemble_stiffness_coarse(LA::MPI::SparseMatrix &    stiffness_matrix,
                            LA::MPI::Vector &          rhs,
                            const DoFHandler<dim> &    dh,
                            AffineConstraints<double> &stiffness_constraints,
                            const FiniteElement<dim> & fe,
                            const Quadrature<dim> &    quadrature,
                            const unsigned int         n_subdivisions) override
  {
    stiffness_matrix = 0;
    if (rhs.size())
      rhs = 0;

    FEValues<dim> fe_values(fe,
                            quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double>                  rhs_values(n_q_points);
    std::vector<double>                  alpha_values(n_q_points);


    const auto lexicographic_to_hierarchic_numbering =
      FETools::lexicographic_to_hierarchic_numbering<dim>(n_subdivisions);


    for (const auto &cell : dh.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        if (rhs.size())
          {
            lod::par.rhs.value_list(fe_values.get_quadrature_points(),
                                    rhs_values);
          }
        Alpha.value_list(fe_values.get_quadrature_points(), alpha_values);

        for (unsigned int c_1 = 0; c_1 < n_subdivisions; ++c_1)
          for (unsigned int c_0 = 0; c_0 < n_subdivisions; ++c_0)
            for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
              for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                {
                  const unsigned int q_index =
                    (c_0 * 2 + q_0) + (c_1 * 2 + q_1) * (2 * n_subdivisions);

                  for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
                    for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
                      {
                        const unsigned int i =
                          lexicographic_to_hierarchic_numbering
                            [(c_0 + i_0) + (c_1 + i_1) * (n_subdivisions + 1)];

                        for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                          for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                            {
                              const unsigned int j =
                                lexicographic_to_hierarchic_numbering
                                  [(c_0 + j_0) +
                                   (c_1 + j_1) * (n_subdivisions + 1)];

                              cell_matrix(i, j) +=
                                // alpha[vector_cell_index] *
                                alpha_values[q_index] *
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