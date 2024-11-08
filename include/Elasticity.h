#ifndef dealii_elasticity_h
#define dealii_elasticity_h

#include <LOD.h>

template <int dim, int spacedim = dim>
class ElasticityProblem : public LOD<dim, spacedim>
{
public:
  ElasticityProblem(const LODParameters<dim, spacedim> &par)
    : LOD<dim, spacedim>(par){};

  typedef LOD<dim, spacedim> lod;


protected:
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

    const FEValuesExtractors::Vector displacement(0);

    const unsigned int dofs_per_cell = lod::fe_fine->n_dofs_per_cell();
    const unsigned int n_q_points    = lod::quadrature_fine->size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<Vector<double>>          rhs_values(n_q_points,
                                           Vector<double>(spacedim));
    const auto                           lexicographic_to_hierarchic_numbering =
      FETools::lexicographic_to_hierarchic_numbering<dim>(
        lod::par.n_subdivisions);

    for (const auto &cell : dh.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        if (rhs.size())
          {
            lod::par.rhs.vector_value_list(fe_values.get_quadrature_points(),
                                           rhs_values);
          }

        for (unsigned int c_1 = 0; c_1 < lod::par.n_subdivisions; ++c_1)
          for (unsigned int c_0 = 0; c_0 < lod::par.n_subdivisions; ++c_0)
            for (unsigned int d_0 = 0; d_0 < 2; ++d_0)
              for (unsigned int d_1 = 0; d_1 < 2; ++d_1)
                for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
                  for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                    {
                      const unsigned int q =
                        (c_0 * 2 + q_0) +
                        (c_1 * 2 + q_1) * (2 * lod::par.n_subdivisions);

                      for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
                        for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
                          {
                            const unsigned int i =
                              lod::fe_fine->component_to_system_index(
                                d_0,
                                lexicographic_to_hierarchic_numbering
                                  [(c_0 + i_0) +
                                   (c_1 + i_1) *
                                     (lod::par.n_subdivisions + 1)]);

                            for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                              for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                                {
                                  const unsigned int j =
                                    lod::fe_fine->component_to_system_index(
                                      d_1,
                                      lexicographic_to_hierarchic_numbering
                                        [(c_0 + j_0) +
                                         (c_1 + j_1) *
                                           (lod::par.n_subdivisions + 1)]);

                                  cell_matrix(i, j) +=
                                    (2 * scalar_product(
                                           fe_values[displacement]
                                             .symmetric_gradient(i, q),
                                           fe_values[displacement]
                                             .symmetric_gradient(j, q)) +
                                     fe_values[displacement].divergence(i, q) *
                                       fe_values[displacement].divergence(j,
                                                                          q)) *
                                    fe_values.JxW(q);
                                }
                            // assemble rhs
                            if (rhs.size())
                              {
                                const auto comp_i =
                                  lod::fe_fine->system_to_component_index(i)
                                    .first;

                                cell_rhs(i) += fe_values.shape_value(i, q) *
                                               rhs_values[q][comp_i] *
                                               fe_values.JxW(q);
                              }
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