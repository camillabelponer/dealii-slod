// LOD as triple-matrix multiplication:
//   A^LOD = C^T A^FEM C
//
// implemented as block system
//
// | A^LOD  0 |
// |   0    0 |
//                =
// | 0  C^T | | 0    0   | | 0 0 |
// | 0   0  | | 0  A^FEM | | C 0 |
//
// via cell loop
//
// A^LOD = ∑ C_i^T A_i^FEM C_i
//
// Needed: C (computed column-wise, for AffineConstraints
// we need to access the full row)

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "util.h"

using namespace dealii;



namespace Step96
{
  struct Parameters
  {
    std::string  physics               = "elasticity";
    unsigned int n_subdivisions_fine   = 4;
    unsigned int n_components          = 1;
    unsigned int n_oversampling        = 2;
    unsigned int n_subdivisions_coarse = 8;
    bool         LOD_stabilization     = true;
    bool         plot_basis            = false;
    bool         compute_error         = true;

    void
    parse(const std::string file_name)
    {
      dealii::ParameterHandler prm;
      prm.add_parameter("physics", physics);
      prm.add_parameter("n subdivisions fine", n_subdivisions_fine);
      prm.add_parameter("n oversampling", n_oversampling);
      prm.add_parameter("n subdivisions coarse", n_subdivisions_coarse);
      prm.add_parameter("LOD stabilization", LOD_stabilization);
      prm.add_parameter("compute error", compute_error);
      prm.parse_input(file_name, "", true);
    }
  };

  namespace LODConstraints
  {
    // We have computed the constraint matrix column-by-column. However, we
    // need the constraint matrix row-by-row during assembly. Communicate
    // the relevant data.
    void
    communicate_constraints(AffineConstraints<double> &constraints,
                            const IndexSet            &locally_owned_dofs,
                            const IndexSet &constraints_to_make_consistent,
                            const MPI_Comm  mpi_communicator)
    {
      using Entries = std::vector<std::pair<types::global_dof_index, double>>;
      using ConstraintLine = std::pair<types::global_dof_index, Entries>;

      std::vector<Entries> constraints_local(locally_owned_dofs.n_elements());

      {
        const auto &local_lines = constraints.get_local_lines();

        const auto [owners_of_my_constraints, _] =
          Utilities::MPI::compute_index_owner_and_requesters(locally_owned_dofs,
                                                             local_lines,
                                                             mpi_communicator);

        std::map<unsigned int, std::vector<ConstraintLine>> send_data;

        for (const auto &line : constraints.get_lines())
          {
            const auto index = line.index;
            const auto rank =
              owners_of_my_constraints[local_lines.index_within_set(index)];
            send_data[rank].emplace_back(index, line.entries);
          }

        const std::map<unsigned int, std::vector<ConstraintLine>> recv_data =
          Utilities::MPI::some_to_some(mpi_communicator, send_data);

        for (const auto &[_, index_and_entries] : recv_data)
          for (const auto &[index, entries] : index_and_entries)
            {
              const auto local_index =
                locally_owned_dofs.index_within_set(index);

              constraints_local[local_index].insert(
                constraints_local[local_index].end(),
                entries.begin(),
                entries.end());
            }
      }

      {
        const auto [_, constrained_indices_by_ranks] =
          Utilities::MPI::compute_index_owner_and_requesters(
            locally_owned_dofs,
            constraints_to_make_consistent,
            mpi_communicator);

        std::map<unsigned int, std::vector<ConstraintLine>> send_data;

        for (const auto &[r, indices] : constrained_indices_by_ranks)
          for (const auto j : indices)
            send_data[r].emplace_back(
              j, constraints_local[locally_owned_dofs.index_within_set(j)]);

        const std::map<unsigned int, std::vector<ConstraintLine>> recv_data =
          Utilities::MPI::some_to_some(mpi_communicator, send_data);


        constraints.clear();

        IndexSet constraints_to_make_consistent_extended =
          constraints_to_make_consistent;

        for (const auto &[_, index_and_entries] : recv_data)
          for (const auto &[_, entries] : index_and_entries)
            for (const auto &[entry, _] : entries)
              constraints_to_make_consistent_extended.add_index(entry);

        constraints.reinit(locally_owned_dofs,
                           constraints_to_make_consistent_extended);

        for (const auto &[_, index_and_entries] : recv_data)
          for (const auto &[index, entries] : index_and_entries)
            constraints.add_constraint(index, entries);
      }
    }

    void
    distribute_local_to_global(
      const AffineConstraints<double>            &constraints,
      const FullMatrix<double>                   &cell_matrix_fem,
      const Vector<double>                       &cell_rhs_fem,
      const std::vector<types::global_dof_index> &local_dof_indices,
      TrilinosWrappers::SparseMatrix             &A_lod,
      LinearAlgebra::distributed::Vector<double> &rhs_lod)
    {
      if (false)
        {
          constraints.distribute_local_to_global(cell_matrix_fem,
                                                 local_dof_indices,
                                                 local_dof_indices,
                                                 A_lod);
        }
      else
        {
          std::set<unsigned int> dofs;
          for (const auto i : local_dof_indices)
            if (constraints.is_constrained(i))
              for (const auto &[j, _] : *constraints.get_constraint_entries(i))
                dofs.insert(j);

          std::vector<unsigned int> v(dofs.begin(), dofs.end());

          FullMatrix<double> C(local_dof_indices.size(), dofs.size());

          for (unsigned int ii = 0; ii < local_dof_indices.size(); ++ii)
            {
              const unsigned int i = local_dof_indices[ii];

              if (constraints.is_constrained(i))
                for (const auto &[j, weight] :
                     *constraints.get_constraint_entries(i))
                  C[ii][std::distance(
                    v.begin(), std::lower_bound(v.begin(), v.end(), j))] =
                    weight;
            }

          FullMatrix<double> CAC(dofs.size(), dofs.size());
          CAC.triple_product(cell_matrix_fem, C, C, true);

          AffineConstraints<double>().distribute_local_to_global(CAC,
                                                                 v,
                                                                 v,
                                                                 A_lod);
        }

      constraints.distribute_local_to_global(cell_rhs_fem,
                                             local_dof_indices,
                                             rhs_lod);
    }


    void
    distribute(const AffineConstraints<double>                  &constraints,
               LinearAlgebra::distributed::Vector<double>       &vec_fem,
               const LinearAlgebra::distributed::Vector<double> &vec_lod)
    {
      const bool has_ghost_elements = vec_lod.has_ghost_elements();

      if (has_ghost_elements == false)
        vec_lod.update_ghost_values();

      for (const auto i : constraints.get_locally_owned_indices())
        if (const auto constraint_entries =
              constraints.get_constraint_entries(i))
          {
            double new_value = 0.0;
            for (const auto &[j, weight] : *constraint_entries)
              new_value += weight * vec_lod[j];

            vec_fem[i - vec_lod.size()] = new_value;
          }

      if (has_ghost_elements == false)
        vec_lod.zero_out_ghost_values();
    }
  } // namespace LODConstraints

  template <int dim>
  class LODProblem
  {
  public:
    LODProblem(const Parameters &params)
      : n_subdivisions_fine(params.n_subdivisions_fine)
      , n_components(params.n_components)
      , n_oversampling(params.n_oversampling)
      , n_subdivisions_coarse(params.n_subdivisions_coarse)
      , LOD_stabilization(params.LOD_stabilization)
      , plot_basis(params.plot_basis)
      , compute_error(params.compute_error)
      , comm(MPI_COMM_WORLD)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
      , repetitions(dim, n_subdivisions_coarse)
      , mapping(1)
      , fe(FE_Q_iso_Q1<dim>(n_subdivisions_fine), n_components)
      , quadrature(QGauss<1>(2), n_subdivisions_fine)
      , patch(n_subdivisions_fine, repetitions, n_components)
      , tria(comm,
             Triangulation<dim>::none,
             true,
             parallel::shared::Triangulation<dim>::partition_custom_signal)
      , timer_output(pcout,
                     dealii::TimerOutput::never,
                     dealii::TimerOutput::wall_times)
    {
      AssertThrow(Utilities::MPI::n_mpi_processes(comm) <=
                    n_subdivisions_coarse,
                  ExcNotImplemented());
    }

    ~LODProblem()
    {
      timer_output.print_wall_time_statistics(comm);
    }


    void
    run(const std::function<void(const FEValues<dim> &, FullMatrix<double> &)>
          &assemble_element_stiffness_matrix,
        const std::function<void(const FEValues<dim> &, Vector<double> &)>
                                             &assemble_element_rhs_vector,
        const std::shared_ptr<Function<dim>> &exact_solution)
    {
      this->make_grid();
      this->setup_system();
      this->setup_basis(assemble_element_stiffness_matrix);
      this->assemble_system(assemble_element_stiffness_matrix,
                            assemble_element_rhs_vector);
      this->solve();
      this->output_results(exact_solution);
    }

  private:
    void
    make_grid()
    {
      TimerOutput::Scope timer(timer_output, "make_grid");

      Point<dim> p1;
      Point<dim> p2;

      for (unsigned int d = 0; d < dim; ++d)
        p2[d] = 1.0;

      const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
      const unsigned int stride =
        (repetitions[dim - 1] + n_procs - 1) / n_procs;

      tria.signals.create.connect([&, stride]() {
        for (const auto &cell : tria.active_cell_iterators())
          {
            unsigned int cell_index = cell->active_cell_index();

            for (unsigned int i = 0; i < dim - 1; ++i)
              cell_index /= repetitions[i];

            cell->set_subdomain_id(cell_index / stride);
          }
      });

      GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);
    }

    void
    setup_system()
    {
      TimerOutput::Scope timer(timer_output, "setup_system");

      const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
      const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);
      const unsigned int stride =
        (repetitions[dim - 1] + n_procs - 1) / n_procs;
      unsigned int range_start =
        (my_rank == 0) ? 0 : ((stride * my_rank) * n_subdivisions_fine + 1);
      unsigned int range_end = stride * (my_rank + 1) * n_subdivisions_fine + 1;

      unsigned int face_dofs = n_components;
      for (unsigned int d = 0; d < dim - 1; ++d)
        face_dofs *= repetitions[d] * n_subdivisions_fine + 1;

      types::global_dof_index n_dofs_coarse = n_components;
      types::global_dof_index n_dofs_fine   = n_components;
      for (unsigned int d = 0; d < dim; ++d)
        {
          n_dofs_coarse *= repetitions[d];
          n_dofs_fine *= repetitions[d] * n_subdivisions_fine + 1;
        }

      AssertDimension(n_dofs_coarse, tria.n_active_cells() * n_components);

      locally_owned_dofs_fem = IndexSet(n_dofs_fine);
      locally_owned_dofs_fem.add_range(
        face_dofs *
          std::min(range_start, repetitions[dim - 1] * n_subdivisions_fine + 1),
        face_dofs *
          std::min(range_end, repetitions[dim - 1] * n_subdivisions_fine + 1));

      locally_owned_dofs_lod = IndexSet(n_dofs_coarse);
      IndexSet locally_relevant_dofs_coarse(n_dofs_coarse);

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            // locally owned dofs
            for (unsigned int c = 0; c < n_components; ++c)
              locally_owned_dofs_lod.add_index(
                cell->active_cell_index() * n_components + c);

            // locally relevant dofs
            patch.reinit(cell, n_oversampling * 2);
            for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
              {
                const auto cell_index =
                  patch.create_cell_iterator(tria, cell)->active_cell_index();

                for (unsigned int c = 0; c < n_components; ++c)
                  locally_relevant_dofs_coarse.add_index(
                    cell_index * n_components + c);
              }
          }

      // 2) ininitialize sparsity pattern
      TrilinosWrappers::SparsityPattern sparsity_pattern_A_lod(
        locally_owned_dofs_lod, comm);

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            // A_lod sparsity pattern
            patch.reinit(cell, n_oversampling * 2);
            std::vector<types::global_dof_index> local_dof_indices_coarse;
            for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
              {
                const auto cell_index =
                  patch.create_cell_iterator(tria, cell)->active_cell_index();

                for (unsigned int c = 0; c < n_components; ++c)
                  local_dof_indices_coarse.emplace_back(
                    cell_index * n_components + c);
              }

            for (const auto &row_index : local_dof_indices_coarse)
              sparsity_pattern_A_lod.add_row_entries(row_index,
                                                     local_dof_indices_coarse);
          }

      sparsity_pattern_A_lod.compress();
      A_lod.reinit(sparsity_pattern_A_lod);

      rhs_lod.reinit(locally_owned_dofs_lod,
                     locally_relevant_dofs_coarse,
                     comm);
      solution_lod.reinit(rhs_lod);
    }

    void
    setup_basis(
      const std::function<void(const FEValues<dim> &, FullMatrix<double> &)>
        &assemble_element_stiffness_matrix)
    {
      TimerOutput::Scope timer(timer_output, "setup_basis");

      IndexSet constraints_lod_fem_locally_owned_dofs(
        locally_owned_dofs_fem.size() + locally_owned_dofs_lod.size());
      constraints_lod_fem_locally_owned_dofs.add_indices(
        locally_owned_dofs_lod);
      constraints_lod_fem_locally_owned_dofs.add_indices(
        locally_owned_dofs_fem, locally_owned_dofs_lod.size());

      IndexSet constraints_lod_fem_locally_stored_constraints(
        locally_owned_dofs_fem.size() + locally_owned_dofs_lod.size());
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            patch.reinit(cell, n_oversampling);
            const auto                           n_dofs_patch = patch.n_dofs();
            std::vector<types::global_dof_index> local_dof_indices_fine(
              n_dofs_patch);
            patch.get_dof_indices(local_dof_indices_fine);

            for (const auto &i : local_dof_indices_fine)
              constraints_lod_fem_locally_stored_constraints.add_index(
                i + locally_owned_dofs_lod.size());
          }


      IndexSet temp(locally_owned_dofs_fem.size() +
                    locally_owned_dofs_lod.size());
      temp.add_indices(locally_owned_dofs_lod);
      temp.add_indices(constraints_lod_fem_locally_stored_constraints);

      constraints_lod_fem.reinit(constraints_lod_fem_locally_owned_dofs, temp);

      // 4) set dummy constraints
      LODPatchProblem<dim> lod_patch_problem(n_components,
                                             LOD_stabilization,
                                             fe);

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            patch.reinit(cell, n_oversampling);

            const auto n_dofs_patch = patch.n_dofs();

            AffineConstraints<double> patch_constraints;
            for (unsigned int d = 0; d < 2 * dim; ++d)
              patch.make_zero_boundary_constraints(d, patch_constraints);
            patch_constraints.close();

            TrilinosWrappers::SparsityPattern sparsity_pattern(n_dofs_patch,
                                                               n_dofs_patch);
            patch.create_sparsity_pattern(patch_constraints, sparsity_pattern);
            sparsity_pattern.compress();

            TrilinosWrappers::SparseMatrix patch_stiffness_matrix(
              sparsity_pattern);

            FEValues<dim> fe_values(mapping,
                                    fe,
                                    quadrature,
                                    update_values | update_gradients |
                                      update_JxW_values);

            for (unsigned int cell = 0; cell < patch.n_cells(); ++cell)
              {
                fe_values.reinit(patch.create_cell_iterator(tria, cell));

                const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

                FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

                assemble_element_stiffness_matrix(fe_values, cell_matrix);

                std::vector<types::global_dof_index> indices(dofs_per_cell);
                patch.get_dof_indices_of_cell(cell, indices);

                AffineConstraints<double>().distribute_local_to_global(
                  cell_matrix, indices, patch_stiffness_matrix);
              }

            patch_stiffness_matrix.compress(VectorOperation::values::add);

            const auto selected_basis_function =
              lod_patch_problem.setup_basis(patch,
                                            patch.cell_index(cell),
                                            patch_stiffness_matrix);

            std::vector<types::global_dof_index> local_dof_indices_fine(
              n_dofs_patch);
            patch.get_dof_indices(local_dof_indices_fine);

            for (unsigned int c = 0; c < n_components; ++c)
              for (unsigned int i = 0; i < n_dofs_patch; ++i)
                {
                  const auto index =
                    local_dof_indices_fine[i] + locally_owned_dofs_lod.size();

                  if (constraints_lod_fem.is_constrained(index) == false)
                    constraints_lod_fem.add_line(index);

                  constraints_lod_fem.add_entry(index,
                                                cell->active_cell_index() *
                                                    n_components +
                                                  c,
                                                selected_basis_function[c][i]);
                }
          }

      LODConstraints::communicate_constraints(
        constraints_lod_fem,
        constraints_lod_fem_locally_owned_dofs,
        constraints_lod_fem_locally_stored_constraints,
        comm);

      constraints_lod_fem.close();
    }


    void
    assemble_system(
      const std::function<void(const FEValues<dim> &, FullMatrix<double> &)>
        &assemble_element_stiffness_matrix,
      const std::function<void(const FEValues<dim> &, Vector<double> &)>
        &assemble_element_rhs_vector)
    {
      TimerOutput::Scope timer(timer_output, "assemble_system");

      FEValues<dim> fe_values(mapping,
                              fe,
                              quadrature,
                              update_values | update_gradients |
                                update_JxW_values | update_quadrature_points);

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;

            // a) compute FEM element stiffness matrix
            FullMatrix<double> cell_matrix_fem(n_dofs_per_cell,
                                               n_dofs_per_cell);
            Vector<double>     cell_rhs_fem(n_dofs_per_cell);

            assemble_element_stiffness_matrix(fe_values, cell_matrix_fem);
            assemble_element_rhs_vector(fe_values, cell_rhs_fem);

            // b) assemble into LOD matrix by using constraints
            std::vector<types::global_dof_index> local_dof_indices(
              n_dofs_per_cell);

            patch.reinit(cell, 0);
            patch.get_dof_indices(local_dof_indices, true /*hiearchical*/);
            for (auto &i : local_dof_indices)
              i += rhs_lod.size(); // shifted view

            LODConstraints::distribute_local_to_global(constraints_lod_fem,
                                                       cell_matrix_fem,
                                                       cell_rhs_fem,
                                                       local_dof_indices,
                                                       A_lod,
                                                       rhs_lod);
          }


      A_lod.compress(VectorOperation::values::add);
      rhs_lod.compress(VectorOperation::values::add);
    }

    void
    solve()
    {
      TimerOutput::Scope timer(timer_output, "solve");

      TrilinosWrappers::SolverDirect solver;
      solver.solve(A_lod, solution_lod, rhs_lod);
    }

    void
    output_results(const std::shared_ptr<Function<dim>> &exact_solution)
    {
      TimerOutput::Scope timer(timer_output, "output_results");

      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(
        FESystem<dim>(FE_Q_iso_Q1<dim>(n_subdivisions_fine), n_components));
      compute_renumbering_lex(dof_handler);

      DoFHandler<dim> dof_handler_coarse(tria);
      dof_handler_coarse.distribute_dofs(
        FESystem<dim>(FE_DGQ<dim>(0), n_components));
      compute_renumbering_lex(dof_handler_coarse);

      // convert to FEM solution
      LinearAlgebra::distributed::Vector<double> solution_lod_fine(
        dof_handler.locally_owned_dofs(),
        DoFTools::extract_locally_active_dofs(dof_handler),
        comm);

      LODConstraints::distribute(constraints_lod_fem,
                                 solution_lod_fine,
                                 solution_lod);

      // output LOD and FEM results

      DataOutBase::VtkFlags flags;

      if (dim > 1)
        flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      if (dim == n_components)
        {
          std::vector<std::string> labels(dim, "solution_lod_fine");

          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(
              dim, DataComponentInterpretation::component_is_part_of_vector);

          data_out.add_data_vector(dof_handler,
                                   solution_lod_fine,
                                   labels,
                                   data_component_interpretation);
        }
      else
        {
          data_out.add_data_vector(dof_handler,
                                   solution_lod_fine,
                                   "solution_lod_fine");
        }

      if (dim == n_components)
        {
          std::vector<std::string> labels(dim, "solution_lod_coarse");

          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(
              dim, DataComponentInterpretation::component_is_part_of_vector);

          data_out.add_data_vector(dof_handler_coarse,
                                   solution_lod,
                                   labels,
                                   data_component_interpretation);
        }
      else
        {
          data_out.add_data_vector(solution_lod, "solution_lod_coarse");
        }

      if (exact_solution)
        {
          LinearAlgebra::distributed::Vector<double> solution_exact;
          solution_exact.reinit(solution_lod_fine);

          VectorTools::interpolate(mapping,
                                   dof_handler,
                                   *exact_solution,
                                   solution_exact);

          if (dim == n_components)
            {
              std::vector<std::string> labels(dim, "solution_exact");

              std::vector<
                DataComponentInterpretation::DataComponentInterpretation>
                data_component_interpretation(
                  dim,
                  DataComponentInterpretation::component_is_part_of_vector);

              data_out.add_data_vector(dof_handler,
                                       solution_exact,
                                       labels,
                                       data_component_interpretation);
            }
          else
            {
              data_out.add_data_vector(dof_handler,
                                       solution_exact,
                                       "solution_exact");
            }
        }

      if (plot_basis)
        {
          for (unsigned int i = 0; i < solution_lod.size(); ++i)
            {
              LinearAlgebra::distributed::Vector<double> e_i;
              e_i.reinit(solution_lod);
              if (solution_lod.locally_owned_elements().is_element(i))
                e_i[i] = 1.0;

              LinearAlgebra::distributed::Vector<double> basis_i;
              basis_i.reinit(solution_lod_fine);

              LODConstraints::distribute(constraints_lod_fem, basis_i, e_i);

              if (dim == n_components)
                {
                  std::vector<std::string> labels(
                    dim, "x_basis_" + Utilities::int_to_string(i, 5));

                  std::vector<
                    DataComponentInterpretation::DataComponentInterpretation>
                    data_component_interpretation(
                      dim,
                      DataComponentInterpretation::component_is_part_of_vector);

                  data_out.add_data_vector(dof_handler,
                                           basis_i,
                                           labels,
                                           data_component_interpretation);
                }
              else
                {
                  data_out.add_data_vector(dof_handler,
                                           basis_i,
                                           "x_basis_" +
                                             Utilities::int_to_string(i, 5));
                }
            }
        }

      if (compute_error && exact_solution)
        {
          Vector<double> cell_wise_error;

          solution_lod_fine.update_ghost_values();
          VectorTools::integrate_difference(mapping,
                                            dof_handler,
                                            solution_lod_fine,
                                            *exact_solution,
                                            cell_wise_error,
                                            quadrature,
                                            VectorTools::NormType::L2_norm);
          solution_lod_fine.zero_out_ghost_values();

          const auto error =
            VectorTools::compute_global_error(tria,
                                              cell_wise_error,
                                              VectorTools::NormType::L2_norm);

          pcout << "error (solution): " << error << std::endl;
        }

      pcout << solution_lod.l2_norm() << std::endl;
      pcout << solution_lod_fine.l2_norm() << std::endl;

      Vector<double> ranks(tria.n_active_cells());
      ranks = Utilities::MPI::this_mpi_process(comm);
      data_out.add_data_vector(ranks, "rank");

      data_out.build_patches(mapping, n_subdivisions_fine);

      const std::string file_name = "solution.vtu";

      data_out.write_vtu_in_parallel(file_name, comm);
    }

    const unsigned int n_subdivisions_fine;
    const unsigned int n_components;
    const unsigned int n_oversampling;
    const unsigned int n_subdivisions_coarse;
    const bool         LOD_stabilization;
    const bool         plot_basis;
    const bool         compute_error;

    MPI_Comm           comm;
    ConditionalOStream pcout;

    const std::vector<unsigned int> repetitions;

    const MappingQ<dim>  mapping;
    const FESystem<dim>  fe;
    const QIterated<dim> quadrature;

    Patch<dim> patch;

    parallel::shared::Triangulation<dim> tria;

    IndexSet locally_owned_dofs_fem;
    IndexSet locally_owned_dofs_lod;

    AffineConstraints<double> constraints_lod_fem;

    TrilinosWrappers::SparseMatrix A_lod;

    LinearAlgebra::distributed::Vector<double> rhs_lod;
    LinearAlgebra::distributed::Vector<double> solution_lod;

    TimerOutput timer_output;
  };

} // namespace Step96

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim = 2;

  Step96::Parameters params;
  if (argc > 1)
    params.parse(std::string(argv[1]));

  std::function<void(const FEValues<dim> &, FullMatrix<double> &)>
    assemble_element_stiffness_matrix;

  std::function<void(const FEValues<dim> &, Vector<double> &)>
    assemble_element_rhs_vector;

  assemble_element_rhs_vector = [](const FEValues<dim> &fe_values,
                                   Vector<double>      &cell_rhs_fem) {
    for (const unsigned int q_index : fe_values.quadrature_point_indices())
      for (const unsigned int i : fe_values.dof_indices())
        cell_rhs_fem(i) +=
          (fe_values.shape_value(i, q_index) * 1. * fe_values.JxW(q_index));
  };

  std::shared_ptr<Function<dim>> exact_solution;

  if (params.physics == "diffusion")
    {
      params.n_components = 1;

      const auto lexicographic_to_hierarchic_numbering =
        FETools::lexicographic_to_hierarchic_numbering<dim>(
          params.n_subdivisions_fine);

      assemble_element_stiffness_matrix =
        [&params, lexicographic_to_hierarchic_numbering](
          const FEValues<dim> &fe_values, FullMatrix<double> &cell_matrix) {
          const double alpha_value = 1.0; // TODO: random variable

          for (unsigned int c_1 = 0; c_1 < params.n_subdivisions_fine; ++c_1)
            for (unsigned int c_0 = 0; c_0 < params.n_subdivisions_fine; ++c_0)
              for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
                for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                  {
                    const unsigned int q_index =
                      (c_0 * 2 + q_0) +
                      (c_1 * 2 + q_1) * (2 * params.n_subdivisions_fine);

                    for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
                      for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
                        {
                          const unsigned int i =
                            lexicographic_to_hierarchic_numbering
                              [(c_0 + i_0) +
                               (c_1 + i_1) * (params.n_subdivisions_fine + 1)];

                          for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                            for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                              {
                                const unsigned int j =
                                  lexicographic_to_hierarchic_numbering
                                    [(c_0 + j_0) +
                                     (c_1 + j_1) *
                                       (params.n_subdivisions_fine + 1)];

                                cell_matrix(i, j) +=
                                  alpha_value *
                                  (fe_values.shape_grad(i, q_index) *
                                   fe_values.shape_grad(j, q_index) *
                                   fe_values.JxW(q_index));
                              }
                        }
                  }
        };

      assemble_element_rhs_vector = [](const FEValues<dim> &fe_values,
                                       Vector<double>      &cell_rhs_fem) {
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const auto point = fe_values.quadrature_point(q_index);

            double value = 2.0 * std::pow(numbers::PI, 2.0);

            for (unsigned int d = 0; d < dim; ++d)
              value *= std::sin(point[d] * numbers::PI);

            for (const unsigned int i : fe_values.dof_indices())
              cell_rhs_fem(i) += (fe_values.shape_value(i, q_index) * value *
                                  fe_values.JxW(q_index));
          }
      };

      exact_solution = std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
        [&](const auto &p) {
          double value = 1.0;

          for (unsigned int d = 0; d < dim; ++d)
            value *= std::sin(numbers::PI * p[d]);

          return value;
        });
    }
  else if (params.physics == "diffusion-vector")
    {
      params.n_components = dim;

      const auto lexicographic_to_hierarchic_numbering =
        FETools::lexicographic_to_hierarchic_numbering<dim>(
          params.n_subdivisions_fine);

      assemble_element_stiffness_matrix =
        [&params, lexicographic_to_hierarchic_numbering](
          const FEValues<dim> &fe_values, FullMatrix<double> &cell_matrix) {
          const double alpha_value = 1.0; // TODO: random variable

          // TODO: better solution?
          FESystem<dim> fe(FE_Q_iso_Q1<dim>(params.n_subdivisions_fine), dim);

          const FEValuesExtractors::Vector displacement(0);

          for (unsigned int c_1 = 0; c_1 < params.n_subdivisions_fine; ++c_1)
            for (unsigned int c_0 = 0; c_0 < params.n_subdivisions_fine; ++c_0)

              for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
                for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                  {
                    const unsigned int q =
                      (c_0 * 2 + q_0) +
                      (c_1 * 2 + q_1) * (2 * params.n_subdivisions_fine);
                    for (unsigned int d_0 = 0; d_0 < 2; ++d_0)
                      for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
                        for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
                          {
                            const unsigned int i = fe.component_to_system_index(
                              d_0,
                              lexicographic_to_hierarchic_numbering
                                [(c_0 + i_0) +
                                 (c_1 + i_1) *
                                   (params.n_subdivisions_fine + 1)]);

                            for (unsigned int d_1 = 0; d_1 < 2; ++d_1)
                              for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                                for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                                  {
                                    const unsigned int j =
                                      fe.component_to_system_index(
                                        d_1,
                                        lexicographic_to_hierarchic_numbering
                                          [(c_0 + j_0) +
                                           (c_1 + j_1) *
                                             (params.n_subdivisions_fine + 1)]);

                                    if (d_0 == d_1)
                                      cell_matrix(i, j) +=
                                        alpha_value * (1 + d_0) *
                                        (fe_values.shape_grad(i, q) *
                                         fe_values.shape_grad(j, q) *
                                         fe_values.JxW(q));
                                  }
                          }
                  }
        };
    }
  else if (params.physics == "elasticity")
    {
      params.n_components = dim;

      const auto lexicographic_to_hierarchic_numbering =
        FETools::lexicographic_to_hierarchic_numbering<dim>(
          params.n_subdivisions_fine);

      assemble_element_stiffness_matrix =
        [&params, lexicographic_to_hierarchic_numbering](
          const FEValues<dim> &fe_values, FullMatrix<double> &cell_matrix) {
          const double mu_value     = 1.0; // TODO: random variable
          const double lambda_value = 1.0; // TODO: random variable

          // TODO: better solution?
          FESystem<dim> fe(FE_Q_iso_Q1<dim>(params.n_subdivisions_fine), dim);

          const FEValuesExtractors::Vector displacement(0);

          for (unsigned int c_1 = 0; c_1 < params.n_subdivisions_fine; ++c_1)
            for (unsigned int c_0 = 0; c_0 < params.n_subdivisions_fine; ++c_0)

              for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
                for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                  {
                    const unsigned int q =
                      (c_0 * 2 + q_0) +
                      (c_1 * 2 + q_1) * (2 * params.n_subdivisions_fine);
                    for (unsigned int d_0 = 0; d_0 < 2; ++d_0)
                      for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
                        for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
                          {
                            const unsigned int i = fe.component_to_system_index(
                              d_0,
                              lexicographic_to_hierarchic_numbering
                                [(c_0 + i_0) +
                                 (c_1 + i_1) *
                                   (params.n_subdivisions_fine + 1)]);

                            for (unsigned int d_1 = 0; d_1 < 2; ++d_1)
                              for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                                for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                                  {
                                    const unsigned int j =
                                      fe.component_to_system_index(
                                        d_1,
                                        lexicographic_to_hierarchic_numbering
                                          [(c_0 + j_0) +
                                           (c_1 + j_1) *
                                             (params.n_subdivisions_fine + 1)]);
                                    cell_matrix(i, j) +=
                                      (2 * mu_value *
                                         scalar_product(
                                           fe_values[displacement]
                                             .symmetric_gradient(i, q),
                                           fe_values[displacement]
                                             .symmetric_gradient(j, q)) +
                                       lambda_value *
                                         fe_values[displacement].divergence(i,
                                                                            q) *
                                         fe_values[displacement].divergence(
                                           j, q)) *
                                      fe_values.JxW(q);
                                  }
                          }
                  }
        };
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  Step96::LODProblem<dim> problem(params);
  problem.run(assemble_element_stiffness_matrix,
              assemble_element_rhs_vector,
              exact_solution);
}