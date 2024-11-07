
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

using namespace dealii;

template <int dim>
const Table<2, bool>
create_bool_dof_mask(const FiniteElement<dim> &fe,
                     const Quadrature<dim> &   quadrature)
{
  const auto compute_scalar_bool_dof_mask = [&quadrature](const auto &fe) {
    Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);
    MappingQ1<dim> mapping;
    FEValues<dim>  fe_values(mapping, fe, quadrature, update_values);

    Triangulation<dim> tria;
    GridGenerator::hyper_cube(tria);

    fe_values.reinit(tria.begin());
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
        {
          double sum = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            sum += fe_values.shape_value(i, q) * fe_values.shape_value(j, q);
          if (sum != 0)
            bool_dof_mask(i, j) = true;
        }

    return bool_dof_mask;
  };

  Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);

  if (fe.n_components() == 1)
    {
      bool_dof_mask = compute_scalar_bool_dof_mask(fe);
    }
  else
    {
      const auto scalar_bool_dof_mask =
        compute_scalar_bool_dof_mask(fe.base_element(0));

      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
          if (scalar_bool_dof_mask[fe.system_to_component_index(i).second]
                                  [fe.system_to_component_index(j).second])
            bool_dof_mask[i][j] = true;
    }


  return bool_dof_mask;
};

template <int dim, int spacedim>
void
test()
{
  const unsigned int fe_degree = 3;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);

  FESystem<dim>        fe(FE_Q_iso_Q1<dim>(fe_degree), spacedim);
  const QIterated<dim> quadrature(QGauss<1>(2), fe_degree);

  Table<2, bool> bool_dof_mask = create_bool_dof_mask(fe, quadrature);


  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values);

  fe_values.reinit(tria.begin());
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix_1(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_2(dofs_per_cell, dofs_per_cell);

  const FEValuesExtractors::Vector displacement(0);

  {
    for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            cell_matrix_1(i, j) +=
              // (fe_values.shape_grad(i, q_index) *
              //  fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
              (2 * scalar_product(
                     fe_values[displacement].symmetric_gradient(i, q_index),
                     fe_values[displacement].symmetric_gradient(j, q_index)) +
               fe_values[displacement].divergence(i, q_index) *
                 fe_values[displacement].divergence(j, q_index)) *
              fe_values.JxW(q_index);
      }

    cell_matrix_1.print_formatted(std::cout, 3, false, 10);
    std::cout << std::endl;
  }

  {
    const auto lexicographic_to_hierarchic_numbering =
      FETools::lexicographic_to_hierarchic_numbering<dim>(fe_degree);

    if (dim == 1)
      {
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
            if (bool_dof_mask[i][j])
              for (const unsigned int q_index :
                   fe_values.quadrature_point_indices())
                {
                  cell_matrix_2(i, j) +=
                    (fe_values.shape_grad(i, q_index) *
                     fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
                }
      }
    else if (dim == 2)
      {
        // for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        //   for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
        //     if (bool_dof_mask[i][j])
        //       for (const unsigned int q_index :
        //            fe_values.quadrature_point_indices())
        for (unsigned int c_1 = 0; c_1 < fe_degree * 2; ++c_1)
          for (unsigned int c_0 = 0; c_0 < fe_degree * 2; ++c_0)
            for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
              for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
                  for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
                    for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                      for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                        {
                          const unsigned int q_index =
                            (c_0 * 2 + q_0) + (c_1 * 2 + q_1) * (2 * fe_degree);
                          const unsigned int i =
                            lexicographic_to_hierarchic_numbering
                              [(c_0 + i_0) + (c_1 + i_1) * (fe_degree + 1)];
                          const unsigned int j =
                            lexicographic_to_hierarchic_numbering
                              [(c_0 + j_0) + (c_1 + j_1) * (fe_degree + 1)];

                          cell_matrix_2(i, j) +=
                            // (fe_values.shape_grad(i, q_index) *
                            //  fe_values.shape_grad(j, q_index) *
                            //  fe_values.JxW(q_index));
                            (2 * scalar_product(
                                   fe_values[displacement].symmetric_gradient(
                                     i, q_index),
                                   fe_values[displacement].symmetric_gradient(
                                     j, q_index)) +
                             fe_values[displacement].divergence(i, q_index) *
                               fe_values[displacement].divergence(j, q_index)) *
                            fe_values.JxW(q_index);
                        }
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
    std::cout << "new assemble" << std::endl;
    cell_matrix_2.print_formatted(std::cout, 3, false, 10);
    std::cout << std::endl;

    double diff = -1;
    for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
      for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
        diff = std::max(diff, abs(cell_matrix_1(i, j) - cell_matrix_2(i, j)));
    std::cout << diff << std::endl;
  }
}

int
main()
{
  // test<1, 2>();
  test<2, 2>();
}