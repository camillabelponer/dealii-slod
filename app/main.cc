#include <slod.h>

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  std::string                      prm_file;
  if (argc > 1)
    prm_file = argv[1];
  else
    prm_file = "parameters.prm";
  
  SLODParameters<2, 2> par;
  SLOD<2>                          slod(par);
  ParameterAcceptor::initialize(prm_file);

  slod.run();
  
}
