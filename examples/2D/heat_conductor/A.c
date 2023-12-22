#include <string.h>
#include <gmshc.h>

int main(int argc, char **argv)
{
    // Before using any functions in the C API, Gmsh must be initialized.
    // In the C API the last argument of all functions returns the error
    // code, if any.
    int ierr;
    gmshInitialize(argc, argv, 1, 0, &ierr);
    gmshOptionSetNumber("General.Terminal", 1, &ierr);
    gmshModelAdd("square", &ierr);

    // Define geometric variables.
    const double lc = 10e-1; // characteristic length
    const double L = 10.;    // length of the square
    const double l = 2.;     // length of the Dirichlet boundary

    // Build geomtry of the model.
    // add points
    gmshModelGeoAddPoint(0, 0, 0, lc, 1, &ierr);
    gmshModelGeoAddPoint(L - l, 0, 0, lc, 2, &ierr);
    gmshModelGeoAddPoint(L, 0, 0, lc, 3, &ierr);
    gmshModelGeoAddPoint(L, L - l, 0, lc, 4, &ierr);
    gmshModelGeoAddPoint(L, L, 0, lc, 5, &ierr);
    gmshModelGeoAddPoint(l, L, 0, lc, 6, &ierr);
    gmshModelGeoAddPoint(0, L, 0, lc, 7, &ierr);
    gmshModelGeoAddPoint(0, l, 0, lc, 8, &ierr);
    // add curves
    gmshModelGeoAddLine(1, 2, 1, &ierr);
    gmshModelGeoAddLine(2, 3, 2, &ierr);
    gmshModelGeoAddLine(3, 4, 3, &ierr);
    gmshModelGeoAddLine(4, 5, 4, &ierr);
    gmshModelGeoAddLine(5, 6, 5, &ierr);
    gmshModelGeoAddLine(6, 7, 6, &ierr);
    gmshModelGeoAddLine(7, 8, 7, &ierr);
    gmshModelGeoAddLine(8, 1, 8, &ierr);
    // add curve loops
    const int cl1[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const size_t cl1_n = sizeof(cl1) / sizeof(int);
    gmshModelGeoAddCurveLoop(cl1, cl1_n, 1, 0, &ierr);
    // add surfaces
    const int s1[] = {1};
    const size_t s1_n = sizeof(s1) / sizeof(int);
    gmshModelGeoAddPlaneSurface(s1, s1_n, 1, &ierr);
    // synchronize model
    gmshModelGeoSynchronize(&ierr);

    // Set physical groups.
    const int Dirichlet_1[] = {2, 6};
    const int Dirichlet_2[] = {4, 8};
    const int Neumann[] = {1, 3, 5, 7};
    const int Domain[] = {1};
    gmshModelGeoAddPhysicalGroup(1, Dirichlet_1, 2, 1, "Dirichlet_1", &ierr);
    gmshModelGeoAddPhysicalGroup(1, Dirichlet_2, 2, 2, "Dirichlet_2", &ierr);
    gmshModelGeoAddPhysicalGroup(1, Neumann, 4, 3, "Neumann", &ierr);
    gmshModelGeoAddPhysicalGroup(2, Domain, 1, 4, "Domain", &ierr);
    gmshModelGeoSynchronize(&ierr);

    // Set transfinite meshing algorithm.
    const int cornerTags[] = {1, 3, 5, 7};
    gmshModelGeoMeshSetTransfiniteSurface(1, "Left", cornerTags, 4, &ierr);
    gmshModelGeoMeshSetRecombine(2, 1, 45.0, &ierr);

    // Generate mesh.
    gmshModelGeoSynchronize(&ierr);
    gmshModelMeshGenerate(2, &ierr);
    gmshWrite("A.msh", &ierr);

    // Finalize Gmsh.
    gmshFinalize(&ierr);
    return 0;
}
