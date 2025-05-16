### Diffusion (LOD)

```
mpirun -np 1 ./tests/assembly_02.debug/assembly_02.debug ../input/diffusion_nostab.json
./Elasticity ../input/diffusion_nostab.prm
```

Looks right!

### Diffusion (SLOD)

```
mpirun -np 1 ./tests/assembly_02.debug/assembly_02.debug ../input/diffusion_stab.json
./Elasticity ../input/diffusion_stab.prm
```

In the new implementation, there is an asymmetry, which indicates that there is a bug in the SLOD code.

### Elasticity (LOD)

```
mpirun -np 1 ./tests/assembly_02.debug/assembly_02.debug ../input/elasticity_nostab.json
./Elasticity ../input/elasticity_nostab.prm
```

Looks fine. Absolute magnitude differ by a few percentage.
