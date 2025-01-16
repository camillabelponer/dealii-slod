#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

using namespace dealii;

// namespace internal
// {
//   namespace AffineConstraintsImplementation
//   {
//     template void
//     set_zero_all(
//       const std::vector<types::global_dof_index>                      &cm,
//       LinearAlgebra::distributed::Vector<float, MemorySpace::Default> &vec);

//     template void
//     set_zero_all(
//       const std::vector<types::global_dof_index>                       &cm,
//       LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &vec);
//   } // namespace AffineConstraintsImplementation
// }

namespace dealii::TrilinosWrappers
{
  class MySolverDirect : public SolverDirect
  {
  private:
    /**
     * Actually performs the operations for solving the linear system,
     * including the factorization and forward and backward substitution.
     */
    void
    do_solve()
    {
      // Fetch return value of Amesos Solver functions
      int ierr;

      // First set whether we want to print the solver information to screen or
      // not.
      // ConditionalOStream verbose_cout(std::cout,
      //                                additional_data.output_solver_details);

      // Next allocate the Amesos solver, this is done in two steps, first we
      // create a solver Factory and generate with that the concrete Amesos
      // solver, if possible.
      Amesos Factory;

      // AssertThrow(Factory.Query(additional_data.solver_type.c_str()),
      //             ExcMessage(
      //               std::string("You tried to select the solver type <") +
      //               additional_data.solver_type +
      //               "> but this solver is not supported by Trilinos either "
      //               "because it does not exist, or because Trilinos was not "
      //               "configured for its use."));

      solver.reset(
        Factory.Create(additional_data.solver_type.c_str(), *linear_problem));

      // verbose_cout << "Starting symbolic factorization" << std::endl;
      ierr = solver->SymbolicFactorization();
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));

      // verbose_cout << "Starting numeric factorization" << std::endl;
      ierr = solver->NumericFactorization();
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));

      // verbose_cout << "Starting solve" << std::endl;
      ierr = solver->Solve();
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));

      // Finally, let the deal.II SolverControl object know what has
      // happened. If the solve succeeded, the status of the solver control will
      // turn into SolverControl::success.
      solver_control.check(0, 0);

      if (solver_control.last_check() != SolverControl::success)
        AssertThrow(false,
                    SolverControl::NoConvergence(solver_control.last_step(),
                                                 solver_control.last_value()));
    }

    /**
     * Local dummy solver control object.
     */
    SolverControl solver_control_own;

    /**
     * Reference to the object that controls convergence of the iterative
     * solver. In fact, for these Trilinos wrappers, Trilinos does so itself,
     * but we copy the data from this object before starting the solution
     * process, and copy the data back into it afterwards.
     */
    SolverControl &solver_control;

    /**
     * A structure that collects the Trilinos sparse matrix, the right hand
     * side vector and the solution vector, which is passed down to the
     * Trilinos solver.
     */
    std::unique_ptr<Epetra_LinearProblem> linear_problem;

    /**
     * A structure that contains the Trilinos solver and preconditioner
     * objects.
     */
    std::unique_ptr<Amesos_BaseSolver> solver;

    /**
     * Store a copy of the flags for this particular solver.
     */
    AdditionalData additional_data;

  public:
    /**
     * Constructor. Creates the solver without solver control object.
     */
    explicit MySolverDirect(const AdditionalData &data = AdditionalData());

    /**
     * Constructor. Takes the solver control object and creates the solver.
     */
    MySolverDirect(SolverControl        &cn,
                   const AdditionalData &data = AdditionalData())
      : SolverDirect(cn, data)
      , solver_control(cn)
      , additional_data(data.output_solver_details, data.solver_type){};

    /**
     * Destructor.
     */
    virtual ~MySolverDirect() = default;

    void
    solve(const Epetra_Operator    &A,
          Epetra_MultiVector       &x,
          const Epetra_MultiVector &b)
    {
      linear_problem = std::make_unique<Epetra_LinearProblem>(
        const_cast<Epetra_Operator *>(&A),
        &x,
        const_cast<Epetra_MultiVector *>(&b));
      do_solve();
    }
  };
}; // namespace dealii::TrilinosWrappers


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

void
Gauss_elimination(const FullMatrix<double>             &rhs,
                  const TrilinosWrappers::SparseMatrix &sparse_matrix,
                  FullMatrix<double>                   &solution)
{
  // create preconditioner
  TrilinosWrappers::PreconditionILU ilu;
  ilu.initialize(sparse_matrix);

  Assert(sparse_matrix.m() == sparse_matrix.n(), ExcInternalError());
  Assert(rhs.m() == sparse_matrix.m(), ExcInternalError());
  Assert(rhs.m() == solution.m(), ExcInternalError());
  Assert(rhs.n() == solution.n(), ExcInternalError());

  solution = 0.0;

  const unsigned int n_dofs       = rhs.m();
  const unsigned int Ndofs_coarse = rhs.n();

  const unsigned int n_blocks        = Ndofs_coarse;
  const unsigned int n_blocks_stride = n_blocks;


  for (unsigned int b = 0; b < n_blocks; b += n_blocks_stride)
    {
      const unsigned int bend = std::min(n_blocks, b + n_blocks_stride);

      std::vector<double> rhs_temp(n_dofs * (bend - b));
      std::vector<double> solution_temp(n_dofs * (bend - b));

      for (unsigned int i = 0; i < (bend - b); ++i)
        for (unsigned int j = 0; j < Ndofs_coarse; ++j)
          {
            rhs_temp[i * n_dofs + j] = rhs(i + b, j); // rhs[i + b][j];
            solution_temp[i * n_dofs + j] =
              solution(i + b, j); // solution[i + b][j];
          }

      std::vector<double *> rhs_ptrs(bend - b);
      std::vector<double *> sultion_ptrs(bend - b);

      for (unsigned int i = 0; i < (bend - b); ++i)
        {
          rhs_ptrs[i]     = &rhs_temp[i * n_dofs];      //&rhs[i + b][0];
          sultion_ptrs[i] = &solution_temp[i * n_dofs]; //&solution[i + b][0];
        }

      const Epetra_CrsMatrix &mat  = sparse_matrix.trilinos_matrix();
      const Epetra_Operator  &prec = ilu.trilinos_operator();

      Epetra_MultiVector trilinos_dst(View,
                                      mat.OperatorRangeMap(),
                                      sultion_ptrs.data(),
                                      sultion_ptrs.size());
      Epetra_MultiVector trilinos_src(View,
                                      mat.OperatorDomainMap(),
                                      rhs_ptrs.data(),
                                      rhs_ptrs.size());

      ReductionControl solver_control(100, 1.e-10, 1.e-6, false, false);

      if (false)
        {
          TrilinosWrappers::SolverCG solver(solver_control);
          solver.solve(mat, trilinos_dst, trilinos_src, prec);
        }
      else
        {
          TrilinosWrappers::MySolverDirect solver(solver_control);
          solver.initialize(sparse_matrix);
          solver.solve(mat, trilinos_dst, trilinos_src);
        }

      for (unsigned int i = 0; i < (bend - b); ++i)
        for (unsigned int j = 0; j < Ndofs_coarse; ++j)
          {
            solution(i + b, j) = solution_temp[i * n_dofs + j];
          }
    }
}


template <int dim>
class Patch
{
public:
  Patch(const unsigned int               fe_degree,
        const std::vector<unsigned int> &repetitions,
        const unsigned int               n_components = 1)
    : fe_degree(fe_degree)
    , n_components(n_components)
    , dofs_per_cell(Utilities::pow(fe_degree + 1, dim) * n_components)
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

  bool
  at_boundary(const unsigned int surface) const
  {
    const unsigned int d = surface / 2; // direction
    const unsigned int s = surface % 2; // left or right surface

    if (s == 0)
      return patch_subdivions_start[d] == 0;
    else // (s == 1)
      return this->repetitions[d] ==
             patch_subdivions_start[d] + this->patch_size[d];
  }

  unsigned int
  n_dofs() const
  {
    unsigned int n_dofs_patch = n_components;
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
        const auto cc   = c / n_components;
        const auto comp = cc % n_components;

        auto indices = index_to_indices<dim>(cc, patch_dofs);

        for (unsigned int d = 0; d < dim; ++d)
          indices[d] += patch_subdivions_start[d];

        // TODO: check!
        dof_indices[(hiarchical ? lexicographic_to_hierarchic_numbering[cc] :
                                  cc) *
                      n_components +
                    comp] =
          indices_to_index<dim>(indices, global_dofs) * n_components + comp;
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
          constraints.add_line(i0);
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

  unsigned int
  cell_index(
    const typename Triangulation<dim>::active_cell_iterator &cell) const
  {
    for (unsigned int i = 0; i < n_cells(); ++i)
      if (create_cell_iterator(cell->get_triangulation(), i) == cell)
        return i;

    return numbers::invalid_unsigned_int;
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

        const unsigned int index_c =
          indices_to_index<dim>(indices_1, patch_dofs);

        for (unsigned int cc = 0; cc < n_components; ++cc) // TODO: check!
          dof_indices[lexicographic_to_hierarchic_numbering[c] * n_components +
                      cc] = index_c * n_components + cc;
      }
  }


  template <typename Number, typename SparsityPatternType>
  void
  create_sparsity_pattern(const AffineConstraints<Number> &constraints,
                          SparsityPatternType             &dsp) const
  {
    for (unsigned int cell = 0; cell < this->n_cells(); ++cell)
      {
        std::vector<types::global_dof_index> indices(this->dofs_per_cell);
        this->get_dof_indices_of_cell(cell, indices);

        constraints.add_entries_local_to_global(indices, dsp);
      }
  }

  void
  get_dofs_vectors(std::vector<unsigned int> &all_dofs,
                   std::vector<unsigned int> &internal_dofs,
                   std::vector<unsigned int> &internal_boundary_dofs,
                   std::vector<unsigned int> &domain_boundary_dofs) const
  {
    all_dofs.clear();
    internal_dofs.clear();
    internal_boundary_dofs.clear();
    domain_boundary_dofs.clear();

    for (unsigned int id = 0; id < n_dofs(); ++id)
      all_dofs.push_back(id);

    AssertDimension(dim, 2);
    unsigned int N_boundary_dofs = 4 * fe_degree;
    for (auto id = N_boundary_dofs; id < n_dofs(); ++id)
      internal_dofs.push_back(id);

    std::set<unsigned int> internal_bd_set;
    std::set<unsigned int> domain_bd_set;


    for (unsigned int surface = 0; surface < 2 * dim; ++surface)
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
              if (at_boundary(surface))
                domain_bd_set.insert(i0);
              else
                internal_bd_set.insert(i0);
            }
      }
    internal_boundary_dofs.assign(internal_bd_set.begin(),
                                  internal_bd_set.end());
    domain_boundary_dofs.assign(domain_bd_set.begin(), domain_bd_set.end());

    // corners that are the intersection of a surface at the boundary and an
    // internal surface should be still part of internal_boundary_idx, while
    // it doesn't really matter if they are still in domain_boundary_idx
  }

private:
  const unsigned int        fe_degree;
  const unsigned int        n_components;
  const unsigned int        dofs_per_cell;
  std::vector<unsigned int> lexicographic_to_hierarchic_numbering;

  std::array<unsigned int, dim> repetitions;
  std::array<unsigned int, dim> patch_start;
  std::array<unsigned int, dim> patch_size;
  std::array<unsigned int, dim> patch_subdivions_start;
  std::array<unsigned int, dim> patch_subdivions_size;
};


template <int dim>
const Table<2, bool>
create_bool_dof_mask(const FiniteElement<dim> &fe,
                     const Quadrature<dim>    &quadrature)
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
