#include <slod.h>

int main(int argc, char* argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  SLOD<2> slod(MPI_COMM_WORLD);
  slod.make_grid();
  slod.make_fe();
  slod.create_patches();
  slod.compute_basis_function_candidates();
  // slod.stabilize();
  // slod.assemble_global_matrix();
}
