#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/lapack_full_matrix.h>

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
      if (solver_control)
        {
          solver_control->check(0, 0);

          if (solver_control->last_check() != SolverControl::success)
            AssertThrow(
              false,
              SolverControl::NoConvergence(solver_control->last_step(),
                                           solver_control->last_value()));
        }
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
    SmartPointer<SolverControl> solver_control;

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
    explicit MySolverDirect(const AdditionalData &data = AdditionalData())
      : SolverDirect(data)
    {}

    /**
     * Constructor. Takes the solver control object and creates the solver.
     */
    MySolverDirect(SolverControl        &cn,
                   const AdditionalData &data = AdditionalData())
      : SolverDirect(cn, data)
      , solver_control(&cn)
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

    void
    solve(const TrilinosWrappers::SparseMatrix &sparse_matrix,
          FullMatrix<double>                   &solution,
          const FullMatrix<double>             &rhs)
    {
      Assert(sparse_matrix.m() == sparse_matrix.n(), ExcInternalError());
      Assert(rhs.m() == sparse_matrix.m(), ExcInternalError());
      Assert(rhs.m() == solution.m(), ExcInternalError());
      Assert(rhs.n() == solution.n(), ExcInternalError());

      solution = 0.0;

      const unsigned int m = rhs.m();
      const unsigned int n = rhs.n();

      FullMatrix<double> rhs_t(n, m);
      FullMatrix<double> solution_t(n, m);

      rhs_t.copy_transposed(rhs);
      solution_t.copy_transposed(solution);

      std::vector<double *> rhs_ptrs(n);
      std::vector<double *> sultion_ptrs(n);

      for (unsigned int i = 0; i < n; ++i)
        {
          rhs_ptrs[i]     = &rhs_t[i][0];
          sultion_ptrs[i] = &solution_t[i][0];
        }

      const Epetra_CrsMatrix &mat = sparse_matrix.trilinos_matrix();

      Epetra_MultiVector trilinos_dst(View,
                                      mat.OperatorRangeMap(),
                                      sultion_ptrs.data(),
                                      sultion_ptrs.size());
      Epetra_MultiVector trilinos_src(View,
                                      mat.OperatorDomainMap(),
                                      rhs_ptrs.data(),
                                      rhs_ptrs.size());

      this->initialize(sparse_matrix);
      this->solve(mat, trilinos_dst, trilinos_src);

      solution.copy_transposed(solution_t);
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

              return a.first < b.first;
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
const Table<2, bool>
create_bool_dof_mask_Q_iso_Q1(const FiniteElement<dim> &fe,
                              const Quadrature<dim>    &quadrature)
{
  Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);

  if (fe.n_components() == 1)
    {
      MappingQ1<dim> mapping;
      FEValues<dim>  fe_values(mapping,
                              fe,
                              quadrature,
                              update_values | update_gradients);

      Triangulation<dim> tria;
      GridGenerator::hyper_cube(tria);

      fe_values.reinit(tria.begin());

      const unsigned int n_subdivisions = fe.degree;

      const auto lexicographic_to_hierarchic_numbering =
        FETools::lexicographic_to_hierarchic_numbering<dim>(n_subdivisions);

      for (unsigned int c_1 = 0; c_1 < n_subdivisions; ++c_1)
        for (unsigned int c_0 = 0; c_0 < n_subdivisions; ++c_0)

          for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
            for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
              {
                const unsigned int i =
                  lexicographic_to_hierarchic_numbering[(c_0 + i_0) +
                                                        (c_1 + i_1) *
                                                          (n_subdivisions + 1)];

                for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                  for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                    {
                      const unsigned int j =
                        lexicographic_to_hierarchic_numbering
                          [(c_0 + j_0) + (c_1 + j_1) * (n_subdivisions + 1)];

                      double sum = 0;

                      for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
                        for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                          {
                            const unsigned int q_index =
                              (c_0 * 2 + q_0) +
                              (c_1 * 2 + q_1) * (2 * n_subdivisions);

                            sum += fe_values.shape_grad(i, q_index) *
                                   fe_values.shape_grad(j, q_index);
                          }
                      if (sum != 0)
                        bool_dof_mask(i, j) = true;
                    }
              }
    }
  else
    {
      const auto scalar_bool_dof_mask =
        create_bool_dof_mask_Q_iso_Q1(fe.base_element(0), quadrature);

      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
          if (scalar_bool_dof_mask[fe.system_to_component_index(i).second]
                                  [fe.system_to_component_index(j).second])
            bool_dof_mask[i][j] = true;
    }

  return bool_dof_mask;
}



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



template <int dim>
class Patch
{
public:
  Patch(const unsigned int               fe_degree,
        const std::vector<unsigned int> &repetitions,
        const unsigned int               n_components = 1)
    : fe(FE_Q_iso_Q1<dim>(fe_degree), n_components)
    , fe_degree(fe_degree)
    , n_components(n_components)
    , dofs_per_cell(Utilities::pow(fe_degree + 1, dim) * n_components)
    , lexicographic_to_hierarchic_numbering(
        FETools::lexicographic_to_hierarchic_numbering<dim>(fe_degree))
    , bool_dof_mask_Q_iso_Q1(
        create_bool_dof_mask_Q_iso_Q1(fe,
                                      QIterated<dim>(QGauss<1>(2), fe_degree)))
  {
    for (unsigned int d = 0; d < dim; ++d)
      this->repetitions[d] = repetitions[d];
  }

  /**
   * Initialize patch given of the index of the left-bottom cell
   * and the the patch size in each direction.
   */
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

  /**
   * Initialize patch given a cell and the number of layers around the
   * cell.
   */
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

  /**
   * Return how many cells the patch contrains.
   */
  unsigned int
  n_cells() const
  {
    unsigned int n_cells = 1;
    for (const auto i : patch_size)
      n_cells *= i;

    return n_cells;
  }

  /**
   * Create a cell iterator to the n-th cell in the patch.
   */
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

  /**
   * Return the index of the cell within the patch.
   */
  unsigned int
  cell_index(
    const typename Triangulation<dim>::active_cell_iterator &cell) const
  {
    for (unsigned int i = 0; i < n_cells(); ++i)
      if (create_cell_iterator(cell->get_triangulation(), i) == cell)
        return i;

    return numbers::invalid_unsigned_int;
  }

  /**
   * Return if patch is the boundary.
   */
  bool
  at_boundary(const unsigned int surface) const
  {
    const unsigned int d = surface / 2; // direction
    const unsigned int s = surface % 2; // left or right surface

    if (s == 0)
      return patch_start[d] == 0;
    else // (s == 1)
      return this->repetitions[d] == patch_start[d] + this->patch_size[d];
  }

  /**
   * Return the number of degrees of freedom on a patch.
   */
  unsigned int
  n_dofs() const
  {
    unsigned int n_dofs_patch = n_components;
    for (const auto i : patch_subdivions_size)
      n_dofs_patch *= i + 1;

    return n_dofs_patch;
  }

  /**
   * Return global dof indices of unknown on a patch.
   */
  void
  get_dof_indices(std::vector<types::global_dof_index> &dof_indices,
                  const bool hiarchical = false) const
  {
    AssertDimension(dof_indices.size(), this->n_dofs());

    Assert((hiarchical == false) || n_cells() == 1, ExcInternalError());

    auto patch_dofs = patch_subdivions_size;
    for (auto &i : patch_dofs)
      i += 1;

    auto global_dofs = repetitions;
    for (auto &i : global_dofs)
      i = i * fe_degree + 1;

    for (unsigned int c = 0; c < this->n_dofs(); ++c)
      {
        const auto cc   = c / n_components;
        const auto comp = c % n_components;

        auto indices = index_to_indices<dim>(cc, patch_dofs);

        for (unsigned int d = 0; d < dim; ++d)
          indices[d] += patch_subdivions_start[d];

        dof_indices[hiarchical ?
                      fe.component_to_system_index(
                        comp, lexicographic_to_hierarchic_numbering[cc]) :
                      (cc * n_components + comp)] =
          indices_to_index<dim>(indices, global_dofs) * n_components + comp;
      }
  }

  /**
   * Return local dof indices of a cell within the patch.
   */
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

        for (unsigned int cc = 0; cc < n_components; ++cc)
          dof_indices[fe.component_to_system_index(
            cc, lexicographic_to_hierarchic_numbering[c])] =
            index_c * n_components + cc;
      }
  }

  /**
   * Make zero boundary constraints for a specified face.
   */
  template <typename Number>
  void
  make_zero_boundary_constraints(const unsigned int         surface,
                                 AffineConstraints<Number> &constraints) const
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

          for (unsigned int c = 0; c < n_components; ++c)
            constraints.add_line(i0 * n_components + c);
        }
  }

  /**
   * Create sparsity pattern for a patch-local system matrix.
   */
  template <typename Number, typename SparsityPatternType>
  void
  create_sparsity_pattern(const AffineConstraints<Number> &constraints,
                          SparsityPatternType             &dsp) const
  {
    for (unsigned int cell = 0; cell < this->n_cells(); ++cell)
      {
        std::vector<types::global_dof_index> indices(this->dofs_per_cell);
        this->get_dof_indices_of_cell(cell, indices);

        constraints.add_entries_local_to_global(indices,
                                                dsp,
                                                true,
                                                bool_dof_mask_Q_iso_Q1);
      }
  }

  /**
   * Parition Dofss into patch-internal DoFs, patch-boundary DoFs, and
   * domain-boundary DoFs.
   */
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
                for (unsigned int c = 0; c < n_components; ++c)
                  domain_bd_set.insert(i0 * n_components + c);
              else
                for (unsigned int c = 0; c < n_components; ++c)
                  internal_bd_set.insert(i0 * n_components + c);
            }
      }
    internal_boundary_dofs.assign(internal_bd_set.begin(),
                                  internal_bd_set.end());
    domain_boundary_dofs.assign(domain_bd_set.begin(), domain_bd_set.end());

    for (const auto id : all_dofs)
      if ((internal_bd_set.find(id) == internal_bd_set.end()) &&
          (domain_bd_set.find(id) == domain_bd_set.end()))
        internal_dofs.push_back(id);

    // corners that are the intersection of a surface at the boundary and an
    // internal surface should be still part of internal_boundary_idx, while
    // it doesn't really matter if they are still in domain_boundary_idx
  }

private:
  const FESystem<dim> fe;

  const unsigned int              fe_degree;
  const unsigned int              n_components;
  const unsigned int              dofs_per_cell;
  const std::vector<unsigned int> lexicographic_to_hierarchic_numbering;
  const Table<2, bool>            bool_dof_mask_Q_iso_Q1;

  std::array<unsigned int, dim> repetitions;
  std::array<unsigned int, dim> patch_start;
  std::array<unsigned int, dim> patch_size;
  std::array<unsigned int, dim> patch_subdivions_start;
  std::array<unsigned int, dim> patch_subdivions_size;
};



template <int dim>
class LODPatchProblem
{
public:
  LODPatchProblem(const unsigned int   n_components,
                  const bool           LOD_stabilization,
                  const FiniteElement<dim> &fe)
    : n_components(n_components)
    , LOD_stabilization(LOD_stabilization)
    , fe(fe)
  {}

  std::vector<Vector<double>>
  setup_basis(const Patch<dim>               &patch,
              const unsigned int              central_cell_id,
              TrilinosWrappers::SparseMatrix &patch_stiffness_matrix)
  {
    const auto n_dofs_patch = patch.n_dofs();

    AffineConstraints<double> patch_constraints;
    for (unsigned int d = 0; d < 2 * dim; ++d)
      patch.make_zero_boundary_constraints(d, patch_constraints);
    patch_constraints.close();

    std::vector<Vector<double>> selected_basis_function(
      n_components, Vector<double>(n_dofs_patch));

    const unsigned int N_dofs_coarse = patch.n_cells() * n_components;
    const unsigned int N_dofs_fine   = n_dofs_patch;

    FullMatrix<double> PT(N_dofs_fine, N_dofs_coarse);
    FullMatrix<double> P_Ainv_PT(N_dofs_coarse);
    FullMatrix<double> Ainv_PT(N_dofs_fine, N_dofs_coarse);
    // SLOD matrices
    std::vector<unsigned int>            internal_dofs_fine;
    std::vector<unsigned int>            all_dofs_fine; // to be filled
    std::vector<unsigned int> /*patch_*/ boundary_dofs_fine;
    std::vector<unsigned int>            domain_boundary_dofs_fine;

    patch.get_dofs_vectors(all_dofs_fine,
                           internal_dofs_fine,
                           /*patch_*/ boundary_dofs_fine,
                           domain_boundary_dofs_fine);

    std::vector<unsigned int> all_dofs_coarse(all_dofs_fine.begin(),
                                              all_dofs_fine.begin() +
                                                N_dofs_coarse);

    unsigned int       considered_candidates = N_dofs_coarse - 1;
    const unsigned int N_boundary_dofs       = boundary_dofs_fine.size();
    const unsigned int N_internal_dofs       = internal_dofs_fine.size();

    FullMatrix<double> PT_boundary(N_boundary_dofs, N_dofs_coarse);
    FullMatrix<double> S_boundary(N_boundary_dofs, N_internal_dofs);

    // ... by looping over cells in patch
    for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
      {
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        std::vector<types::global_dof_index> indices(dofs_per_cell);
        patch.get_dof_indices_of_cell(cell, indices);

        for (unsigned int ii = 0; ii < indices.size(); ++ii)
          {
            const auto i = indices[ii];

            const double scale =
              (ii < 4 * fe.n_dofs_per_vertex()) ?
                0.25 :
                ((ii < 4 * fe.n_dofs_per_vertex() + 4 * fe.n_dofs_per_line()) ?
                   0.5 :
                   1.0);

            PT[i][cell * n_components +
                  fe.system_to_component_index(ii).first] = scale;
          }
      }

    if (LOD_stabilization && boundary_dofs_fine.size() > 0)
      {
        PT_boundary.extract_submatrix_from(PT,
                                           boundary_dofs_fine,
                                           all_dofs_coarse);
        if (true)
          S_boundary.extract_submatrix_from(patch_stiffness_matrix,
                                            boundary_dofs_fine,
                                            internal_dofs_fine);
        else
          {
            for (unsigned int row_id = 0; row_id < boundary_dofs_fine.size();
                 ++row_id)
              for (unsigned int col_id = 0; col_id < internal_dofs_fine.size();
                   ++col_id)
                S_boundary.set(
                  row_id,
                  col_id,
                  patch_stiffness_matrix.el(boundary_dofs_fine[row_id],
                                            internal_dofs_fine[col_id]));
          }
      }

    for (unsigned int i = 0; i < N_dofs_coarse; ++i)
      {
        for (const auto j : boundary_dofs_fine)
          PT(j, i) = 0.0;
        for (const auto j : domain_boundary_dofs_fine)
          PT(j, i) = 0.0;
      }

    for (const auto j : boundary_dofs_fine)
      patch_stiffness_matrix.clear_row(j, 1);
    for (const auto j : domain_boundary_dofs_fine)
      patch_stiffness_matrix.clear_row(j, 1);

#if false
            TrilinosWrappers::SolverDirect solver;
#else
    TrilinosWrappers::MySolverDirect solver;
#endif
    solver.solve(patch_stiffness_matrix, Ainv_PT, PT);

    PT.Tmmult(P_Ainv_PT, Ainv_PT);
    P_Ainv_PT.gauss_jordan();

    Vector<double> e_i(N_dofs_coarse);
    Vector<double> triple_product_inv_e_i(N_dofs_coarse);

    if (!LOD_stabilization || (boundary_dofs_fine.size() == 0))
      // LOD
      // also in the case of : oversampling == 0 ||
      // or if the patch is the whole domain
      {
        for (unsigned int c = 0; c < n_components; ++c)
          {
            e_i                                     = 0.0;
            e_i[central_cell_id * n_components + c] = 1.0;
            P_Ainv_PT.vmult(triple_product_inv_e_i, e_i);
            Ainv_PT.vmult(selected_basis_function[c], triple_product_inv_e_i);
          }
      }
    else // SLOD
      {
        FullMatrix<double>       BD(N_boundary_dofs, N_dofs_coarse);
        FullMatrix<double>       B_full(N_boundary_dofs, N_dofs_coarse);
        LAPACKFullMatrix<double> SVD(considered_candidates,
                                     considered_candidates);
        FullMatrix<double> Ainv_PT_internal(N_internal_dofs, N_dofs_coarse);

        Vector<double> internal_selected_basis_function(N_internal_dofs);
        Vector<double> c_i(N_internal_dofs);
        internal_selected_basis_function = 0.0;

        for (unsigned int c = 0; c < n_components; ++c)
          selected_basis_function[c] = 0.0;

        Ainv_PT_internal.extract_submatrix_from(Ainv_PT,
                                                internal_dofs_fine,
                                                all_dofs_coarse);
        S_boundary.mmult(B_full, Ainv_PT_internal);

        // creating the matrix B_full using all components from all
        // candidates
        PT_boundary *= -1;
        B_full.mmult(BD, P_Ainv_PT);
        PT_boundary.mmult(BD, P_Ainv_PT, true);

        for (unsigned int d = 0; d < n_components; ++d)
          {
            Vector<double> B_d0(N_boundary_dofs);

            for (unsigned int i = 0; i < N_boundary_dofs; ++i)
              B_d0[i] = BD(i, central_cell_id * n_components + d);

            Vector<double> d_i(considered_candidates);
            Vector<double> BDTBD0(considered_candidates);
            d_i    = 0;
            BDTBD0 = 0;

            // std::vector<unsigned int> other_phi(all_dofs_fine.begin()
            // + 1,
            //                                     all_dofs_fine.begin()
            //                                     +
            //                                       N_dofs_coarse);
            std::vector<unsigned int> other_phi(all_dofs_fine.begin(),
                                                all_dofs_fine.begin() +
                                                  N_dofs_coarse);
            other_phi.erase(other_phi.begin() + central_cell_id * n_components +
                            d);

            {
              FullMatrix<double> newBD(N_boundary_dofs, considered_candidates);
              FullMatrix<double> BDTBD(considered_candidates,
                                       considered_candidates);

              Assert(
                other_phi.size() == considered_candidates,
                ExcNotImplemented(
                  "inconsistent number of candidates basis function on the patch"));
              std::vector<unsigned int> boundary_dofs_vector_temp(
                all_dofs_fine.begin(), all_dofs_fine.begin() + N_boundary_dofs);

              newBD.extract_submatrix_from(BD,
                                           boundary_dofs_vector_temp,
                                           other_phi);

              newBD.Tmmult(BDTBD, newBD);

              newBD.Tvmult(BDTBD0, B_d0);

              SVD.copy_from(BDTBD);
            }

            SVD.compute_inverse_svd(1e-15); // stores U V as normal, but
                                            // 1/singular_value_i
            d_i = 0.0;
            SVD.vmult(d_i, BDTBD0);
            d_i *= -1;
            auto U  = SVD.get_svd_u();
            auto Vt = SVD.get_svd_vt();

            AssertDimension(SVD.m(), SVD.n());
            AssertDimension(U.m(), U.n());
            AssertDimension(Vt.m(), Vt.n());
            AssertDimension(U.m(), Vt.n());
            AssertDimension(U.m(), SVD.n());
            AssertDimension(U.m(), considered_candidates);

            for (int i = (considered_candidates - 1); i >= 0; --i)
              {
                if (d_i.linfty_norm() < 0.5)
                  break;
                Vector<double> uT(considered_candidates);
                Vector<double> v(considered_candidates);
                // for (auto j : all_dofs_coarse)
                for (unsigned int j = 0; j < considered_candidates; ++j)
                  {
                    uT[j] = U(j, i);
                    v[j]  = Vt(i, j);
                  }
                FullMatrix<double> vuT(considered_candidates,
                                       considered_candidates);
                // do uT scalar BDTBD0 first
                vuT.outer_product(v, uT);
                Vector<double> correction(d_i.size());
                vuT.vmult(correction, BDTBD0);
                correction *= // Sigma_minus1(i, i); //
                  SVD.singular_value(i);

                d_i += correction;
              }

            Vector<double> DeT(N_dofs_coarse);
            e_i                                     = 0.0;
            e_i[central_cell_id * n_components + d] = 1.0;
            P_Ainv_PT.vmult(DeT, e_i);
            c_i = DeT;

            // for (unsigned int index = 0; index <
            // considered_candidates;
            // ++index)
            for (unsigned int index = 0; index < other_phi.size(); ++index)
              {
                e_i                   = 0.0;
                e_i[other_phi[index]] = 1.0;

                P_Ainv_PT.vmult(DeT, e_i);

                DeT *= d_i[index];

                c_i += DeT;
              }

            Ainv_PT_internal.vmult(internal_selected_basis_function, c_i);

            // somehow the following does not work
            // internal_selected_basis_function.extract_subvector_to(internal_selected_basis_function.begin(),
            // internal_selected_basis_function.end(),
            // selected_basis_function.begin()+N_boundary_dofs);
            for (unsigned int id = 0; id < internal_dofs_fine.size(); ++id)
              selected_basis_function[d][internal_dofs_fine[id]] =
                internal_selected_basis_function[id];
          }
      }

    for (unsigned int c = 0; c < n_components; ++c)
      {
        selected_basis_function[c] /= selected_basis_function[c].l2_norm();

        patch_constraints.set_zero(selected_basis_function[c]);
      }

    return selected_basis_function;
  }

private:
  const unsigned int   n_components;
  const bool           LOD_stabilization;
  const FiniteElement<dim> &fe;
};
