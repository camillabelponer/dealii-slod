#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

using namespace dealii;

template <int dim>
void
compute_renumbering_lex(dealii::DoFHandler<dim> &dof_handler)
{
  std::vector<dealii::types::global_dof_index> dof_indices(
    dof_handler.get_fe().n_dofs_per_cell());

  dealii::IndexSet active_dofs;
  dealii::DoFTools::extract_locally_active_dofs(dof_handler, active_dofs);
  const auto partitioner =
    std::make_shared<dealii::Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(), active_dofs, MPI_COMM_WORLD);

  std::vector<std::pair<dealii::types::global_dof_index, dealii::Point<dim>>>
    points_all;

  dealii::FEValues<dim> fe_values(
    dof_handler.get_fe(),
    dealii::Quadrature<dim>(dof_handler.get_fe().get_unit_support_points()),
    dealii::update_quadrature_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned() == false)
        continue;

      fe_values.reinit(cell);

      cell->get_dof_indices(dof_indices);

      for (unsigned int i = 0; i < dof_indices.size(); ++i)
        {
          if (dof_handler.locally_owned_dofs().is_element(dof_indices[i]))
            points_all.emplace_back(dof_indices[i],
                                    fe_values.quadrature_point(i));
        }
    }

  std::sort(points_all.begin(),
            points_all.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  points_all.erase(std::unique(points_all.begin(),
                               points_all.end(),
                               [](const auto &a, const auto &b) {
                                 return a.first == b.first;
                               }),
                   points_all.end());

  std::sort(points_all.begin(),
            points_all.end(),
            [](const auto &a, const auto &b) {
              std::vector<double> a_(dim);
              std::vector<double> b_(dim);

              a.second.unroll(a_.begin(), a_.end());
              std::reverse(a_.begin(), a_.end());

              b.second.unroll(b_.begin(), b_.end());
              std::reverse(b_.begin(), b_.end());

              for (unsigned int d = 0; d < dim; ++d)
                {
                  if (std::abs(a_[d] - b_[d]) > 1e-8 /*epsilon*/)
                    return a_[d] < b_[d];
                }

              return true;
            });

  std::vector<dealii::types::global_dof_index> result(
    dof_handler.n_locally_owned_dofs());

  for (unsigned int i = 0; i < result.size(); ++i)
    {
      result[partitioner->global_to_local(points_all[i].first)] =
        partitioner->local_to_global(i);
    }

  dof_handler.renumber_dofs(result);
}


template <typename Number, unsigned int dim>
void
make_zero_boundary_constraints(
  const unsigned int                   surface,
  const std::array<unsigned int, dim> &n_subdivisions,
  AffineConstraints<Number>           &constraints)
{
  const unsigned int d = surface / 2; // direction
  const unsigned int s = surface % 2; // left or right surface

  unsigned int n0 = 1;
  for (unsigned int i = d + 1; i < dim; ++i)
    n0 *= n_subdivisions[i] + 1;

  unsigned int n1 = 1;
  for (unsigned int i = 0; i < d; ++i)
    n1 *= n_subdivisions[i] + 1;

  const unsigned int n2 = n1 * (n_subdivisions[d] + 1);

  for (unsigned int i = 0; i < n0; ++i)
    for (unsigned int j = 0; j < n1; ++j)
      {
        const unsigned i0 = i * n2 + (s == 0 ? 0 : n_subdivisions[d]) * n1 + j;

        constraints.constrain_dof_to_zero(i0);
      }
}

template <int dim>
class Patch
{
public:
  Patch(const unsigned int               fe_degree,
        const std::vector<unsigned int> &repetitions)
    : fe_degree(fe_degree)
    , lexicographic_to_hierarchic_numbering(
        FETools::lexicographic_to_hierarchic_numbering<dim>(fe_degree))
  {
    for (unsigned int d = 0; d < dim; ++d)
      this->repetitions[d] = repetitions[d];
  }

  void
  reinit(const std::array<unsigned int, dim> &patch_start,
         const std::array<unsigned int, dim> &patch_size)
  {
    this->patch_start = patch_start;
    this->patch_size  = patch_size;


    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_subdivions_start[d] = patch_start[d] * fe_degree;
        patch_subdivions_size[d]  = patch_size[d] * fe_degree;
      }
  }

  void
  reinit(const typename Triangulation<dim>::active_cell_iterator &cell,
         const unsigned int                                       n_overlap)
  {
    const unsigned int index = cell->active_cell_index();
    const unsigned int i     = index % repetitions[0];
    const unsigned int j     = index / repetitions[0];

    std::array<unsigned int, dim> patch_start = {{i, j}};
    std::array<unsigned int, dim> patch_size  = {{1, 1}};

    for (unsigned int d = 0; d < 2 * dim; ++d)
      {
        auto cell_neighbor = cell;

        for (unsigned int i = 0; i < n_overlap; ++i)
          {
            if (cell_neighbor->at_boundary(d) == false)
              {
                if ((d % 2) == 0)
                  patch_start[d / 2]--;
                patch_size[d / 2]++;

                cell_neighbor = cell_neighbor->neighbor(d);
              }
            else
              {
                break;
              }
          }
      }

    this->reinit(patch_start, patch_size);
  }

  unsigned int
  n_dofs() const
  {
    unsigned int n_dofs_patch = 1;
    for (const auto i : patch_subdivions_size)
      n_dofs_patch *= i + 1;

    return n_dofs_patch;
  }

  void
  get_dof_indices(std::vector<types::global_dof_index> &dof_indices) const
  {
    AssertDimension(dof_indices.size(), this->n_dofs());

    for (unsigned int j = 0, c = 0; j <= patch_subdivions_size[1]; ++j)
      for (unsigned int i = 0; i <= patch_subdivions_size[0]; ++i, ++c)
        dof_indices[c] =
          (i + patch_subdivions_start[0]) +
          (j + patch_subdivions_start[1]) * (repetitions[0] * fe_degree + 1);
  }

  template <typename Number>
  void
  make_zero_boundary_constraints(const unsigned int         surface,
                                 AffineConstraints<Number> &constraints)
  {
    ::make_zero_boundary_constraints<Number, dim>(surface,
                                                  patch_subdivions_size,
                                                  constraints);
  }

  unsigned int
  n_cells() const
  {
    return patch_size[0] * patch_size[1];
  }

  typename Triangulation<dim>::active_cell_iterator
  create_cell_iterator(const Triangulation<dim> &tria,
                       const unsigned int        index) const
  {
    const unsigned int i = index % patch_size[0];
    const unsigned int j = index / patch_size[0];

    return tria.create_cell_iterator(
      CellId((patch_start[0] + i) + (patch_start[1] + j) * repetitions[0], {}));
  }

  void
  get_dof_indices_of_cell(
    const unsigned int                    index,
    std::vector<types::global_dof_index> &dof_indices) const
  {
    const unsigned int i = index % patch_size[0];
    const unsigned int j = index / patch_size[0];

    for (unsigned int jj = 0, c = 0; jj <= fe_degree; ++jj)
      for (unsigned int ii = 0; ii <= fe_degree; ++ii, ++c)
        dof_indices[lexicographic_to_hierarchic_numbering[c]] =
          (ii + i * fe_degree) +
          (jj + j * fe_degree) * (patch_subdivions_size[0] + 1);
  }

private:
  const unsigned int        fe_degree;
  std::vector<unsigned int> lexicographic_to_hierarchic_numbering;

  std::array<unsigned int, dim> repetitions;
  std::array<unsigned int, dim> patch_start;
  std::array<unsigned int, dim> patch_size;
  std::array<unsigned int, dim> patch_subdivions_start;
  std::array<unsigned int, dim> patch_subdivions_size;
};

int
main()
{
  // 0) parameters
  const unsigned int dim       = 2;
  const unsigned int fe_degree = 7;
  const unsigned int n_overlap = 3; // numbers::invalid_unsigned_int

  std::vector<unsigned int> repetitions = {{10, 10}};
  Point<dim>                p1(0, 0);
  Point<dim>                p2(1, 1);

  // 1) define patch (parameter)
  std::array<unsigned int, dim> patch_start = {{1, 4}};
  std::array<unsigned int, dim> patch_size  = {{4, 3}};

  // 2) create mesh
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);

  // 3) create patch
  Patch<dim> patch(fe_degree, repetitions);

  if (n_overlap == numbers::invalid_unsigned_int)
    patch.reinit(patch_start, patch_size);
  else
    patch.reinit(tria.create_cell_iterator(CellId(
                   patch_start[0] + patch_start[1] * repetitions[0], {})),
                 n_overlap);


  // 4) make constraints on patch
  AffineConstraints<double> patch_constraints;
  for (unsigned int d = 0; d < 2 * dim; ++d)
    patch.make_zero_boundary_constraints<double>(d, patch_constraints);
  patch_constraints.close();

  // 5) assemble system on patch
  const auto         n_dofs_patch = patch.n_dofs();
  FullMatrix<double> A(n_dofs_patch, n_dofs_patch);
  Vector<double>     rhs(n_dofs_patch);
  Vector<double>     solution(n_dofs_patch);

  FE_Q_iso_Q1<dim>   fe(fe_degree);
  const QIterated<2> quadrature(QGauss<1>(2), fe_degree);
  FEValues<2>        fe_values(fe,
                        quadrature,
                        update_values | update_gradients | update_JxW_values);

  // ... by looping over cells in patch
  for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
    {
      fe_values.reinit(patch.create_cell_iterator(tria, cell));

      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) *
                 fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));

          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) +=
              (fe_values.shape_value(i, q_index) * 1. * fe_values.JxW(q_index));
        }

      std::vector<types::global_dof_index> indices(dofs_per_cell);
      patch.get_dof_indices_of_cell(cell, indices);

      patch_constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, indices, A, rhs);
    }

  // 6) solve patch system
  A.gauss_jordan();
  A.vmult(solution, rhs);

  // 7) visualization on fine mesh
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  compute_renumbering_lex(dof_handler);

  Vector<double> solution_fine(dof_handler.n_dofs());

  std::vector<types::global_dof_index> patch_indices(n_dofs_patch);
  patch.get_dof_indices(patch_indices);
  AffineConstraints<double>().distribute_local_to_global(solution,
                                                         patch_indices,
                                                         solution_fine);

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;

  MappingQ<dim> mapping(1);

  DataOut<dim> data_out;
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(solution_fine, "solution");

  data_out.build_patches(mapping, fe_degree);

  const std::string file_name = "solution.vtu";

  std::ofstream file(file_name);
  data_out.write_vtu(file);

  solution_fine.print(std::cout);
}
