
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

using namespace dealii;

template <int dim>
void
test()
{
  const unsigned int fe_degree = 3;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);

  FE_Q_iso_Q1<dim>     fe(fe_degree);
  const QIterated<dim> quadrature(QGauss<1>(2), fe_degree);


  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values);

  fe_values.reinit(tria.begin());

  {
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            cell_matrix(i, j) +=
              (fe_values.shape_grad(i, q_index) *
               fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
      }

    cell_matrix.print_formatted(std::cout, 3, false, 10);
    std::cout << std::endl;
  }

  {
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    const auto lexicographic_to_hierarchic_numbering =
      FETools::lexicographic_to_hierarchic_numbering<dim>(fe_degree);

    if (dim == 1)
      {
        for (unsigned int c_0 = 0; c_0 < fe_degree; ++c_0)
          for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
            for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
              for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                {
                  const unsigned int q_index = (c_0 * 2 + q_0);
                  const unsigned int i =
                    lexicographic_to_hierarchic_numbering[(c_0 + i_0)];
                  const unsigned int j =
                    lexicographic_to_hierarchic_numbering[(c_0 + j_0)];

                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q_index) *
                     fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
                }
      }
    else if (dim == 2)
      {
        for (unsigned int c_1 = 0; c_1 < fe_degree; ++c_1)
          for (unsigned int c_0 = 0; c_0 < fe_degree; ++c_0)
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

                          cell_matrix(i, j) +=
                            (fe_values.shape_grad(i, q_index) *
                             fe_values.shape_grad(j, q_index) *
                             fe_values.JxW(q_index));
                        }
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }

    cell_matrix.print_formatted(std::cout, 3, false, 10);
    std::cout << std::endl;
  }
}

int
main()
{
  test<1>();
  test<2>();
}