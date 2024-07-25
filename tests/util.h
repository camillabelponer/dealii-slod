#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

using namespace dealii;


template <int dim>
std::array<unsigned int, dim>
index_to_indices(const unsigned int                  index,
                 const std::array<unsigned int, dim> Ns)
{
  std::array<unsigned int, dim> indices;

  if (dim >= 1)
    indices[0] = index % Ns[0];

  if (dim >= 2)
    indices[1] = (index / Ns[0]) % Ns[1];

  if (dim >= 3)
    indices[2] = index / (Ns[0] * Ns[1]);

  return indices;
}


template <int dim>
std::array<unsigned int, dim>
index_to_indices(const unsigned int index, const unsigned int N)
{
  std::array<unsigned int, dim> Ns;
  std::fill(Ns.begin(), Ns.end(), N);
  return index_to_indices<dim>(index, Ns);
}


template <int dim>
unsigned int
indices_to_index(const std::array<unsigned int, dim> indices,
                 const std::array<unsigned int, dim> Ns)
{
  unsigned int index = 0;

  if (dim >= 1)
    index += indices[0];

  if (dim >= 2)
    index += indices[1] * Ns[0];

  if (dim >= 3)
    index += indices[2] * Ns[0] * Ns[1];

  return index;
}


template <int dim>
unsigned int
indices_to_index(const std::array<unsigned int, dim> index,
                 const unsigned int                  N)
{
  std::array<unsigned int, dim> Ns;
  std::fill(Ns.begin(), Ns.end(), N);
  return indices_to_index<dim>(index, Ns);
}

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


template <int dim>
class Patch
{
public:
  Patch(const unsigned int               fe_degree,
        const std::vector<unsigned int> &repetitions)
    : fe_degree(fe_degree)
    , dofs_per_cell(Utilities::pow(fe_degree + 1, dim))
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
    auto patch_start =
      index_to_indices<dim>(cell->active_cell_index(), repetitions);

    std::array<unsigned int, dim> patch_size;
    std::fill(patch_size.begin(), patch_size.end(), 1);

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
  get_dof_indices(std::vector<types::global_dof_index> &dof_indices,
                  const bool hiarchical = false) const
  {
    AssertDimension(dof_indices.size(), this->n_dofs());

    auto patch_dofs = patch_subdivions_size;
    for (auto &i : patch_dofs)
      i += 1;

    auto global_dofs = repetitions;
    for (auto &i : global_dofs)
      i = i * fe_degree + 1;

    for (unsigned int c = 0; c < this->n_dofs(); ++c)
      {
        auto indices = index_to_indices<dim>(c, patch_dofs);

        for (unsigned int d = 0; d < dim; ++d)
          indices[d] += patch_subdivions_start[d];

        dof_indices[hiarchical ? lexicographic_to_hierarchic_numbering[c] : c] =
          indices_to_index<dim>(indices, global_dofs);
      }
  }

  template <typename Number>
  void
  make_zero_boundary_constraints(const unsigned int         surface,
                                 AffineConstraints<Number> &constraints)
  {
    const unsigned int d = surface / 2; // direction
    const unsigned int s = surface % 2; // left or right surface

    unsigned int n0 = 1;
    for (unsigned int i = d + 1; i < dim; ++i)
      n0 *= patch_subdivions_size[i] + 1;

    unsigned int n1 = 1;
    for (unsigned int i = 0; i < d; ++i)
      n1 *= patch_subdivions_size[i] + 1;

    const unsigned int n2 = n1 * (patch_subdivions_size[d] + 1);

    for (unsigned int i = 0; i < n0; ++i)
      for (unsigned int j = 0; j < n1; ++j)
        {
          const unsigned i0 =
            i * n2 + (s == 0 ? 0 : patch_subdivions_size[d]) * n1 + j;

          constraints.constrain_dof_to_zero(i0);
        }
  }

  unsigned int
  n_cells() const
  {
    unsigned int n_cells = 1;
    for (const auto i : patch_size)
      n_cells *= i;

    return n_cells;
  }

  typename Triangulation<dim>::active_cell_iterator
  create_cell_iterator(const Triangulation<dim> &tria,
                       const unsigned int        index) const
  {
    auto indices = index_to_indices<dim>(index, patch_size);

    for (unsigned int d = 0; d < dim; ++d)
      indices[d] += patch_start[d];

    return tria.create_cell_iterator(
      CellId(indices_to_index<dim>(indices, repetitions), {}));
  }

  void
  get_dof_indices_of_cell(
    const unsigned int                    index,
    std::vector<types::global_dof_index> &dof_indices) const
  {
    const auto indices_0 = index_to_indices<dim>(index, patch_size);

    auto patch_dofs = patch_subdivions_size;
    for (auto &i : patch_dofs)
      i += 1;

    for (unsigned int c = 0; c < Utilities::pow(fe_degree + 1, dim); ++c)
      {
        auto indices_1 = index_to_indices<dim>(c, fe_degree + 1);

        for (unsigned int d = 0; d < dim; ++d)
          indices_1[d] += indices_0[d] * fe_degree;

        dof_indices[lexicographic_to_hierarchic_numbering[c]] =
          indices_to_index<dim>(indices_1, patch_dofs);
      }
  }


  template <typename Number, typename SparsityPatternType>
  void
  create_sparsity_pattern(const AffineConstraints<Number> &constraints,
                          SparsityPatternType &            dsp) const
  {
    for (unsigned int cell = 0; cell < this->n_cells(); ++cell)
      {
        std::vector<types::global_dof_index> indices(this->dofs_per_cell);
        this->get_dof_indices_of_cell(cell, indices);

        constraints.add_entries_local_to_global(indices, dsp);
      }
  }

private:
  const unsigned int        fe_degree;
  const unsigned int        dofs_per_cell;
  std::vector<unsigned int> lexicographic_to_hierarchic_numbering;

  std::array<unsigned int, dim> repetitions;
  std::array<unsigned int, dim> patch_start;
  std::array<unsigned int, dim> patch_size;
  std::array<unsigned int, dim> patch_subdivions_start;
  std::array<unsigned int, dim> patch_subdivions_size;
};
