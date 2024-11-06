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

    const unsigned int dofs_per_cell = lod::fe_fine->n_dofs_per_cell();
    const unsigned int n_q_points    = lod::quadrature_fine->size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    // const FEValuesExtractors::Vector displacement(0);

    std::vector<Vector<double>> rhs_values(n_q_points,
                                           Vector<double>(spacedim));

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

        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int component_i =
              lod::fe_fine->system_to_component_index(i).first;

            for (const unsigned int j : fe_values.dof_indices())
              {
                const unsigned int component_j =
                  lod::fe_fine->system_to_component_index(j).first;
                if (lod::bool_dof_mask[i][j])
                  {
                    for (const unsigned int q_point :
                         fe_values.quadrature_point_indices())
                      {
                        cell_matrix(i, j) +=
                          (fe_values.shape_grad(i, q_point)[component_i] *
                             fe_values.shape_grad(j, q_point)[component_j] +
                           fe_values.shape_grad(i, q_point)[component_j] *
                             fe_values.shape_grad(j, q_point)[component_i] +
                           ((component_i == component_j) ?
                              (fe_values.shape_grad(i, q_point) *
                               fe_values.shape_grad(j, q_point)) :
                              0)) *
                          fe_values.JxW(q_point);

                      }
                  }
              }
          }

        // assemble rhs
        if (rhs.size())
          {
            for (const unsigned int i : fe_values.dof_indices())
              {
                const unsigned int component_i =
                  lod::fe_fine->system_to_component_index(i).first;

                for (const unsigned int q_point :
                     fe_values.quadrature_point_indices())
                  {
                    cell_rhs(i) += fe_values.shape_value(i, q_point) *
                                   rhs_values[q_point][component_i] *
                                   fe_values.JxW(q_point);
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